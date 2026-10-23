"""
Live glass-break detection dashboard.

Runs the microphone -> spectrogram -> CNN pipeline (from live_alarm.py) in a
background thread and serves a real-time web dashboard: a probability gauge, a
rolling chart, an RMS meter, an alert banner, a detections log, and browser +
sound notifications on each detection.

Backend is the Python standard library only (http.server + Server-Sent Events);
the frontend is self-contained vanilla JS (no external CDN), so it works offline.

Run from anywhere:
    python src/dashboard.py --device 4
Then open  http://127.0.0.1:8000  in a browser. Ctrl+C to stop.
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import tensorflow as tf

# Reuse the exact, training-matched spectrogram + constants from the CLI tool.
from live_alarm import make_spectrogram_array, FS, MODEL_PATH

RECORD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings")
os.makedirs(RECORD_DIR, exist_ok=True)

# Continuous recorder state (records raw mic audio until stopped).
REC = {"on": False, "frames": [], "lock": threading.Lock(), "start": 0.0, "sr": FS}


# ======================================================================
# Shared state (updated by the audio worker, read by SSE subscribers)
# ======================================================================

class Hub:
    def __init__(self, maxlog=200):
        self.lock = threading.Lock()
        self.subscribers = []          # list of deque, one per SSE client
        self.detections = deque(maxlen=maxlog)
        self.latest = {"t": None, "prob": 0.0, "rms": 0.0, "status": "starting"}

    def subscribe(self):
        q = deque(maxlen=200)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def publish(self, event):
        with self.lock:
            self.latest = event
            if event.get("status") == "alert":
                self.detections.appendleft({"t": event["t"], "prob": event["prob"]})
            for q in self.subscribers:
                q.append(event)


HUB = Hub()


# ======================================================================
# Audio worker: rolling window -> spectrogram -> model -> publish
# ======================================================================

def resolve_device(spec):
    """Return (index, samplerate) for the input device to use.

    Device indices on Windows are unstable (they shift as Bluetooth/USB mics
    connect), and the liveliest endpoint is often WASAPI rather than MME. So if
    `spec` is not an explicit integer index, briefly probe candidate input
    devices and pick the one that actually carries a signal.
    """
    devs = sd.query_devices()
    apis = sd.query_hostapis()

    def sr_of(i):
        return int(devs[i]["default_samplerate"]) or 44100

    def api_of(i):
        return apis[devs[i]["hostapi"]]["name"]

    if isinstance(spec, str) and spec.lower() == "default":
        i = sd.default.device[0]
        return i, sr_of(i)

    if spec is not None:
        try:
            i = int(spec)
            return i, sr_of(i)
        except (ValueError, TypeError):
            candidates = [i for i, d in enumerate(devs)
                          if d["max_input_channels"] > 0 and str(spec).lower() in d["name"].lower()]
    else:
        candidates = [i for i, d in enumerate(devs) if d["max_input_channels"] > 0]

    print("Probing input devices (make some noise to help selection)...")
    results = []
    for i in candidates:
        try:
            sr = sr_of(i)
            rec = sd.rec(int(sr * 0.6), samplerate=sr, channels=1, dtype="float32", device=i)
            sd.wait()
            peak = float(np.max(np.abs(rec)))
            if 0.0 <= peak < 10.0:  # <10 filters broken exclusive-mode endpoints
                api = api_of(i)
                results.append((i, sr, peak, api))
                print(f"   [{i:>2}] {devs[i]['name'][:34]:34s} {api:16s} peak={peak:.5f}")
        except Exception as e:
            print(f"   [{i:>2}] {devs[i]['name'][:34]:34s} skipped ({str(e)[:24]})")

    if not results:
        return (candidates[0], sr_of(candidates[0])) if candidates else (None, FS)

    # WASAPI / WDM-KS endpoints on this hardware raise host errors when opening a
    # continuous InputStream, so restrict the choice to MME / DirectSound, which
    # stream reliably. Pick the loudest, nudging toward MME (most compatible).
    def usable(api):
        return ("MME" in api) or ("DirectSound" in api)

    pool = [r for r in results if usable(r[3])] or results

    def score(r):
        return r[2] + (0.0005 if "MME" in r[3] else 0.0)
    best = max(pool, key=score)
    return best[0], best[1]


def audio_worker(args, model, device, capture_sr):
    REC["sr"] = capture_sr
    window_len = int(args.window * capture_sr)   # buffer is at the mic's native rate
    ring = np.zeros(window_len, dtype=np.float32)
    lock = threading.Lock()

    def callback(indata, frames, time_info, status):
        nonlocal ring
        mono = indata[:, 0]
        n = len(mono)
        with lock:
            if n >= window_len:
                ring = mono[-window_len:].copy()
            else:
                ring = np.roll(ring, -n)
                ring[-n:] = mono
        # Continuous recorder: capture every frame while active.
        if REC["on"]:
            with REC["lock"]:
                if REC["on"]:
                    REC["frames"].append(mono.copy())

    try:
        stream = sd.InputStream(samplerate=capture_sr, channels=1, dtype="float32",
                                device=device, callback=callback)
        stream.start()
    except Exception as e:
        print(f"Could not open device {device} @ {capture_sr}Hz ({e}); "
              f"falling back to system default input.")
        device, capture_sr = None, FS
        REC["sr"] = FS
        window_len = int(args.window * FS)
        ring = np.zeros(window_len, dtype=np.float32)
        stream = sd.InputStream(samplerate=FS, channels=1, dtype="float32",
                                device=None, callback=callback)
        stream.start()

    try:
        time.sleep(args.window)  # prime the buffer
        while True:
            time.sleep(args.hop)
            with lock:
                window = ring.copy()

            rms = float(np.sqrt(np.mean(window ** 2)))
            stamp = time.strftime("%H:%M:%S")

            if rms < args.silence:
                HUB.publish({"t": stamp, "prob": 0.0, "rms": rms, "status": "silence"})
                continue

            # The model expects 22050 Hz; resample the native-rate window to match.
            if capture_sr != FS:
                y = librosa.resample(window, orig_sr=capture_sr, target_sr=FS)
            else:
                y = window
            arr = make_spectrogram_array(y, FS)
            # Training sorted classes alphabetically (glassbreak=0, other=1), so the
            # sigmoid outputs P(other). Glass-break probability is its complement.
            prob = 1.0 - float(model(arr, training=False)[0][0])
            status = "alert" if prob >= args.threshold else "listening"
            HUB.publish({"t": stamp, "prob": prob, "rms": rms, "status": status})
    finally:
        stream.stop()
        stream.close()


# ======================================================================
# Web frontend (self-contained: no external assets)
# ======================================================================

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Glass-Break Detector — Live</title>
<style>
  :root {
    --bg:#0b0f17; --panel:#141a26; --panel2:#0f1420; --line:#243044;
    --text:#e7ecf3; --muted:#8b97a8; --accent:#38bdf8;
    --ok:#22c55e; --warn:#f59e0b; --alert:#ef4444;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,Arial; }
  header { display:flex; align-items:center; justify-content:space-between;
    padding:16px 24px; border-bottom:1px solid var(--line); }
  header h1 { font-size:16px; margin:0; letter-spacing:.3px; }
  header h1 span { color:var(--muted); font-weight:400; }
  .conn { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:13px; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--muted); }
  .dot.live { background:var(--ok); box-shadow:0 0 0 4px rgba(34,197,94,.15); }
  main { max-width:1100px; margin:0 auto; padding:24px;
    display:grid; grid-template-columns:340px 1fr; gap:20px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:20px; }
  .card h2 { margin:0 0 14px; font-size:12px; text-transform:uppercase;
    letter-spacing:1px; color:var(--muted); font-weight:600; }
  #statusCard { text-align:center; transition:background .2s, border-color .2s; }
  #statusPill { display:inline-block; padding:6px 14px; border-radius:999px;
    font-weight:700; font-size:13px; letter-spacing:.5px; }
  .s-listening { background:rgba(56,189,248,.15); color:var(--accent); }
  .s-silence  { background:rgba(139,151,168,.15); color:var(--muted); }
  .s-alert    { background:rgba(239,68,68,.18); color:var(--alert); }
  #probBig { font-size:64px; font-weight:800; line-height:1.1; margin:14px 0 2px;
    font-variant-numeric:tabular-nums; }
  #probLabel { color:var(--muted); font-size:12px; }
  .gauge { height:10px; background:var(--panel2); border-radius:999px; overflow:hidden; margin-top:16px;
    position:relative; }
  .gauge > i { display:block; height:100%; width:0%; background:linear-gradient(90deg,#22c55e,#f59e0b,#ef4444);
    transition:width .25s; }
  .gauge .thr { position:absolute; top:-3px; bottom:-3px; width:2px; background:#fff; opacity:.5; }
  .row { display:flex; justify-content:space-between; margin-top:14px; font-size:13px; }
  .row .k { color:var(--muted); }
  .ctrl { margin-top:16px; }
  .ctrl label { display:flex; justify-content:space-between; color:var(--muted); font-size:12px; }
  input[type=range] { width:100%; accent-color:var(--accent); }
  #chart { width:100%; height:220px; display:block; }
  .rms { height:8px; background:var(--panel2); border-radius:999px; overflow:hidden; margin-top:6px; }
  .rms > i { display:block; height:100%; width:0%; background:var(--accent); transition:width .2s; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { text-align:left; padding:8px 6px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.5px; }
  td.p { font-variant-numeric:tabular-nums; color:var(--alert); font-weight:700; }
  .empty { color:var(--muted); padding:14px 6px; }
  #banner { position:fixed; left:50%; top:20px; transform:translateX(-50%) translateY(-120%);
    background:var(--alert); color:#fff; padding:14px 28px; border-radius:12px; font-weight:800;
    box-shadow:0 10px 40px rgba(239,68,68,.5); transition:transform .3s; z-index:50; }
  #banner.show { transform:translateX(-50%) translateY(0); }
  .muted { color:var(--muted); }
  @media (max-width:820px){ main{ grid-template-columns:1fr; } }
</style>
</head>
<body>
<div id="banner">🚨 GLASS BREAK DETECTED</div>
<header>
  <h1>Glass-Break Detector <span>· live</span></h1>
  <div class="conn"><span id="dot" class="dot"></span><span id="connText">connecting…</span></div>
</header>
<main>
  <section>
    <div class="card" id="statusCard">
      <h2>Status</h2>
      <span id="statusPill" class="s-silence">SILENCE</span>
      <div id="probBig">0%</div>
      <div id="probLabel">glass-break probability</div>
      <div class="gauge"><i id="gaugeFill"></i><span id="thrMark" class="thr"></span></div>
      <div class="row"><span class="k">RMS energy</span><span id="rmsVal">0.000</span></div>
      <div class="rms"><i id="rmsFill"></i></div>
      <div class="ctrl">
        <label>Alert threshold <span id="thrVal">0.80</span></label>
        <input id="thr" type="range" min="0.30" max="0.99" step="0.01" value="0.80">
      </div>
      <div class="row"><span class="k">Notifications</span>
        <span><button id="notifyBtn" style="background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:4px 10px;cursor:pointer">Enable</button>
        <button id="testBtn" style="background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:4px 10px;cursor:pointer;margin-left:6px">Test</button>
        <label class="muted" style="margin-left:8px"><input type="checkbox" id="soundChk" checked> sound</label></span>
      </div>
    </div>
    <div class="card" style="margin-top:20px">
      <h2>Recorder</h2>
      <div class="row" style="margin-top:0">
        <button id="recBtn" style="background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 16px;cursor:pointer;font-weight:700">● Record</button>
        <span id="recStat" class="muted" style="align-self:center">idle</span>
      </div>
      <div id="recPlay" style="margin-top:14px"></div>
      <p class="muted" style="font-size:12px;margin:12px 0 0">Records the mic continuously until you press Stop, then lets you play it back — a quick way to confirm the mic is actually capturing audio.</p>
    </div>
  </section>
  <section>
    <div class="card">
      <h2>Probability (last 60 s)</h2>
      <canvas id="chart"></canvas>
    </div>
    <div class="card" style="margin-top:20px">
      <h2>Detections</h2>
      <table><thead><tr><th>Time</th><th>Probability</th></tr></thead>
      <tbody id="log"><tr><td class="empty" colspan="2">No detections yet — listening…</td></tr></tbody></table>
    </div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
let threshold = 0.80, soundOn = true, lastAlert = false;
const hist = [];               // {p} recent points
const MAXPTS = 60;

// ---- threshold slider ----
$('thr').addEventListener('input', e => {
  threshold = parseFloat(e.target.value);
  $('thrVal').textContent = threshold.toFixed(2);
  $('thrMark').style.left = (threshold*100)+'%';
});
$('thrMark').style.left = (threshold*100)+'%';
$('soundChk').addEventListener('change', e => soundOn = e.target.checked);
$('testBtn').addEventListener('click', () => {
  const t = new Date().toLocaleTimeString();
  setStatus('alert'); fireAlert(0.97, t);
  setTimeout(()=>setStatus('listening'), 2000);
});

// ---- continuous recorder ----
let recording=false, recTimer=null, recStart=0;
$('recBtn').addEventListener('click', async () => {
  const btn=$('recBtn'), stat=$('recStat');
  if(!recording){
    await fetch('/api/record/start');
    recording=true; recStart=Date.now();
    btn.textContent='■ Stop'; btn.style.background='#ef4444'; btn.style.color='#fff';
    $('recPlay').innerHTML='';
    recTimer=setInterval(()=>{ stat.textContent='● recording '+((Date.now()-recStart)/1000).toFixed(1)+'s'; }, 200);
  } else {
    clearInterval(recTimer);
    stat.textContent='saving…';
    const r=await (await fetch('/api/record/stop')).json();
    recording=false; btn.textContent='● Record'; btn.style.background='var(--panel2)'; btn.style.color='var(--text)';
    const silent = r.peak < 0.002;
    stat.innerHTML = `saved · ${r.seconds}s · peak ${r.peak.toFixed(4)}` +
      (silent ? ' <span style="color:var(--alert)">⚠ silent — mic muted?</span>' : '');
    $('recPlay').innerHTML =
      `<audio controls style="width:100%" src="/recordings/${r.file}"></audio>`+
      `<div class="muted" style="font-size:12px;margin-top:6px">`+
      `<a style="color:var(--accent)" href="/recordings/${r.file}" download>download ${r.file}</a></div>`;
  }
});

// ---- notifications ----
$('notifyBtn').addEventListener('click', () => {
  if ('Notification' in window) Notification.requestPermission().then(p => {
    $('notifyBtn').textContent = p === 'granted' ? 'Enabled' : 'Blocked';
  });
});
function beep(){
  if(!soundOn) return;
  try{
    const ac = new (window.AudioContext||window.webkitAudioContext)();
    const o = ac.createOscillator(), g = ac.createGain();
    o.type='square'; o.frequency.value=880; o.connect(g); g.connect(ac.destination);
    g.gain.setValueAtTime(0.001, ac.currentTime);
    g.gain.exponentialRampToValueAtTime(0.25, ac.currentTime+0.02);
    g.gain.exponentialRampToValueAtTime(0.001, ac.currentTime+0.4);
    o.start(); o.stop(ac.currentTime+0.42);
  }catch(e){}
}
function fireAlert(prob, t){
  $('banner').classList.add('show');
  setTimeout(()=>$('banner').classList.remove('show'), 2500);
  beep();
  if('Notification' in window && Notification.permission==='granted')
    new Notification('🚨 Glass break detected', {body:`Probability ${(prob*100).toFixed(1)}% at ${t}`});
  const tb = $('log');
  if(tb.querySelector('.empty')) tb.innerHTML='';
  const tr=document.createElement('tr');
  tr.innerHTML=`<td>${t}</td><td class="p">${(prob*100).toFixed(1)}%</td>`;
  tb.prepend(tr);
  while(tb.children.length>50) tb.removeChild(tb.lastChild);
}

// ---- canvas chart ----
const cv=$('chart'), cx=cv.getContext('2d');
function resize(){ cv.width=cv.clientWidth*devicePixelRatio; cv.height=cv.clientHeight*devicePixelRatio;
  cx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0); }
addEventListener('resize', resize); resize();
function draw(){
  const w=cv.clientWidth, h=cv.clientHeight, pad=6;
  cx.clearRect(0,0,w,h);
  // grid + threshold line
  cx.strokeStyle='#243044'; cx.lineWidth=1;
  for(let i=0;i<=4;i++){ const y=pad+(h-2*pad)*i/4; cx.beginPath(); cx.moveTo(pad,y); cx.lineTo(w-pad,y); cx.stroke(); }
  const ty=pad+(h-2*pad)*(1-threshold);
  cx.strokeStyle='rgba(239,68,68,.7)'; cx.setLineDash([5,4]); cx.beginPath();
  cx.moveTo(pad,ty); cx.lineTo(w-pad,ty); cx.stroke(); cx.setLineDash([]);
  // series
  if(hist.length>1){
    cx.lineWidth=2; cx.strokeStyle='#38bdf8'; cx.beginPath();
    hist.forEach((d,i)=>{ const x=pad+(w-2*pad)*i/(MAXPTS-1); const y=pad+(h-2*pad)*(1-d.p);
      i?cx.lineTo(x,y):cx.moveTo(x,y); });
    cx.stroke();
    const last=hist[hist.length-1], lx=pad+(w-2*pad)*(hist.length-1)/(MAXPTS-1), ly=pad+(h-2*pad)*(1-last.p);
    cx.fillStyle=last.p>=threshold?'#ef4444':'#38bdf8'; cx.beginPath(); cx.arc(lx,ly,4,0,7); cx.fill();
  }
}
draw();

// ---- SSE ----
function setStatus(s){
  const pill=$('statusPill'), card=$('statusCard');
  pill.className='s-'+s;
  pill.textContent = s==='alert'?'🚨 GLASS BREAK':(s==='silence'?'SILENCE':'LISTENING');
  card.style.borderColor = s==='alert'?'var(--alert)':'var(--line)';
  card.style.background = s==='alert'?'rgba(239,68,68,.08)':'var(--panel)';
}
function connect(){
  const es=new EventSource('/stream');
  es.onopen=()=>{ $('dot').className='dot live'; $('connText').textContent='live'; };
  es.onerror=()=>{ $('dot').className='dot'; $('connText').textContent='reconnecting…'; };
  es.onmessage=ev=>{
    const d=JSON.parse(ev.data);
    const prob=d.prob||0;
    $('probBig').textContent=(prob*100).toFixed(1)+'%';
    $('gaugeFill').style.width=(prob*100)+'%';
    $('rmsVal').textContent=(d.rms||0).toFixed(4);
    $('rmsFill').style.width=Math.min(100,(d.rms||0)*100/0.2)+'%';
    // client-side alert decision against the slider threshold
    const isAlert = d.status!=='silence' && prob>=threshold;
    setStatus(d.status==='silence'?'silence':(isAlert?'alert':'listening'));
    if(isAlert && !lastAlert) fireAlert(prob, d.t);
    lastAlert = isAlert;
    hist.push({p:prob}); while(hist.length>MAXPTS) hist.shift();
    draw();
  };
}
connect();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass  # keep the console clean

    def do_GET(self):
        if self.path.startswith("/stream"):
            return self._stream()
        if self.path.startswith("/api/record/start"):
            return self._rec_start()
        if self.path.startswith("/api/record/stop"):
            return self._rec_stop()
        if self.path.startswith("/recordings/"):
            return self._serve_recording()
        if self.path.startswith("/api/state"):
            body = json.dumps(HUB.latest).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = HUB.subscribe()
        try:
            # send the current snapshot immediately
            self._send_event(HUB.latest)
            last_ping = time.time()
            while True:
                if q:
                    self._send_event(q.popleft())
                else:
                    time.sleep(0.1)
                    if time.time() - last_ping > 15:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping = time.time()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            HUB.unsubscribe(q)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _rec_start(self):
        with REC["lock"]:
            REC["frames"] = []
            REC["start"] = time.time()
            REC["on"] = True
        self._json({"recording": True})

    def _rec_stop(self):
        with REC["lock"]:
            REC["on"] = False
            frames = REC["frames"]
            REC["frames"] = []
            elapsed = time.time() - REC["start"]
        if frames:
            data = np.concatenate(frames).astype(np.float32)
        else:
            data = np.zeros(1, dtype=np.float32)
        sr = REC.get("sr", FS)
        fname = time.strftime("rec_%Y%m%d_%H%M%S.wav")
        sf.write(os.path.join(RECORD_DIR, fname), data, sr)
        rms = float(np.sqrt(np.mean(data ** 2)))
        peak = float(np.max(np.abs(data)))
        self._json({"recording": False, "file": fname,
                    "seconds": round(len(data) / sr, 2), "rms": rms, "peak": peak})

    def _serve_recording(self):
        name = os.path.basename(self.path.split("?")[0])  # prevent path traversal
        path = os.path.join(RECORD_DIR, name)
        if not (name.endswith(".wav") and os.path.isfile(path)):
            return self._json({"error": "not found"}, 404)
        data = open(path, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'inline; filename="{name}"')
        self.end_headers()
        self.wfile.write(data)

    def _send_event(self, event):
        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        self.wfile.flush()


def main():
    p = argparse.ArgumentParser(description="Live glass-break detection dashboard.")
    p.add_argument("--device", default=None,
                   help="Input device: an index, or a name substring (e.g. \"Intel\"). "
                        "If omitted or a name, the liveliest matching mic is auto-selected.")
    p.add_argument("--threshold", type=float, default=0.80)
    p.add_argument("--window", type=float, default=2.0)
    p.add_argument("--hop", type=float, default=1.0)
    p.add_argument("--silence", type=float, default=0.005)
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    print("Loading model...")
    model = tf.keras.models.load_model(args.model)
    print(f"Model loaded: {args.model}")

    device, capture_sr = resolve_device(args.device)
    if device is None:
        print("ERROR: no usable input device found. Check mic permissions / mute.")
        return
    dev_name = sd.query_devices(device)["name"]
    print(f"Using mic: [{device}] {dev_name}  @ {capture_sr} Hz")

    worker = threading.Thread(target=audio_worker, args=(args, model, device, capture_sr), daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"\nDashboard ready → {url}")
    print(f"window={args.window}s  hop={args.hop}s  threshold={args.threshold}  silence={args.silence}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()

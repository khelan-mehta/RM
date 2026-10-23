"""Khelan audio engine.

One background thread opens the microphone and does two things on the stream:
  1. Glass / sound recognition — every `hop` seconds it scores the last
     `window` seconds with the existing CNN (reusing live_alarm's exact,
     training-matched spectrogram; glass prob = 1 - sigmoid).
  2. Double-clap detection — a lightweight transient detector on each audio
     block (no ML; two impulsive peaks within a window → trigger).

It never touches the DB or websockets directly: it calls `on_event(dict)` for
every event, and the app layer decides what to persist, notify, or act on.
"""
import io
import os
import threading
import time

import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Constants and spectrogram recipe mirror src/live_alarm.py / the training set
# (kept local so importing this module does NOT pull in TensorFlow).
FS = 22050
N_MELS = 128
IMG_SIZE = (128, 128)
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT, "models", "glass_detector_v2.keras")


def make_spectrogram_array(y, sr):
    """1-D audio window → (1,128,128,3) float32 array, matching training."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig = plt.figure(figsize=(2, 2))
    librosa.display.specshow(mel_db, sr=sr, cmap="magma")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img).astype(np.float32), axis=0)


def resolve_device(spec):
    """Pick (index, samplerate). spec may be None, an int index, 'default',
    or a name substring. Prefers MME/DirectSound endpoints (stream reliably)."""
    devs = sd.query_devices()
    apis = sd.query_hostapis()
    sr_of = lambda i: int(devs[i]["default_samplerate"]) or 44100
    api_of = lambda i: apis[devs[i]["hostapi"]]["name"]

    if isinstance(spec, str) and spec.lower() == "default":
        i = sd.default.device[0]
        return (i, sr_of(i)) if i is not None and i >= 0 else (None, FS)
    if spec is not None:
        try:
            i = int(spec); return i, sr_of(i)
        except (ValueError, TypeError):
            cands = [i for i, d in enumerate(devs)
                     if d["max_input_channels"] > 0 and str(spec).lower() in d["name"].lower()]
            if not cands:  # named device not present → fall back to auto-select
                cands = [i for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    else:
        cands = [i for i, d in enumerate(devs) if d["max_input_channels"] > 0]

    results = []
    for i in cands:
        try:
            sr = sr_of(i)
            rec = sd.rec(int(sr * 0.4), samplerate=sr, channels=1, dtype="float32", device=i)
            sd.wait()
            peak = float(np.max(np.abs(rec)))
            if 0.0 <= peak < 10.0:
                results.append((i, sr, peak, api_of(i)))
        except Exception:
            continue
    if not results:
        return (cands[0], sr_of(cands[0])) if cands else (None, FS)
    pool = [r for r in results if ("MME" in r[3] or "DirectSound" in r[3])] or results
    best = max(pool, key=lambda r: r[2] + (0.0005 if "MME" in r[3] else 0.0))
    return best[0], best[1]


class AudioEngine:
    def __init__(self, on_event, device=None, threshold=0.80, window=2.0, hop=1.0,
                 silence=0.005, clap_peak=0.18, clap_gap=(0.12, 0.7), clap_refractory=0.15):
        self.on_event = on_event
        self.device_spec = device
        self.threshold = threshold
        self.window = window
        self.hop = hop
        self.silence = silence
        self.clap_peak = clap_peak
        self.clap_gap = clap_gap
        self.clap_refractory = clap_refractory

        self.model = None
        self.capture_sr = FS
        self.device_index = None
        self.device_name = "—"
        self.available = False
        self.clap_enabled = True
        self.latest = {"type": "audio", "status": "starting", "prob": 0.0, "rms": 0.0, "t": None}

        self._ring = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._clap_times = []
        self._last_onset = 0.0
        self._thread = None

    def start(self):
        """Returns immediately; the model loads inside the audio thread so the
        web server can come up without waiting ~30s for TensorFlow."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ---- audio thread ----
    def _run(self):
        self._emit_status("loading-model", 0.0, 0.0)
        import tensorflow as tf
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.device_index, self.capture_sr = resolve_device(self.device_spec)
        if self.device_index is None:
            self.available = False
            self._emit_status("no-mic", 0.0, 0.0)
            return
        try:
            self.device_name = sd.query_devices(self.device_index)["name"]
        except Exception:
            self.device_name = str(self.device_index)

        window_len = int(self.window * self.capture_sr)
        self._ring = np.zeros(window_len, dtype=np.float32)

        def callback(indata, frames, time_info, status):
            mono = indata[:, 0]
            n = len(mono)
            with self._lock:
                if n >= window_len:
                    self._ring = mono[-window_len:].copy()
                else:
                    self._ring = np.roll(self._ring, -n)
                    self._ring[-n:] = mono
            self._detect_clap(mono)

        try:
            stream = sd.InputStream(samplerate=self.capture_sr, channels=1, dtype="float32",
                                    device=self.device_index, callback=callback)
            stream.start()
        except Exception as e:
            self.available = False
            self._emit_status("mic-error", 0.0, 0.0, extra={"error": str(e)})
            return

        self.available = True
        try:
            time.sleep(self.window)
            while not self._stop.is_set():
                time.sleep(self.hop)
                with self._lock:
                    window = self._ring.copy()
                rms = float(np.sqrt(np.mean(window ** 2)))
                if rms < self.silence:
                    self._emit_status("silence", 0.0, rms)
                    continue
                y = librosa.resample(window, orig_sr=self.capture_sr, target_sr=FS) \
                    if self.capture_sr != FS else window
                arr = make_spectrogram_array(y, FS)
                prob = 1.0 - float(self.model(arr, training=False)[0][0])
                status = "alert" if prob >= self.threshold else "listening"
                self._emit_status(status, prob, rms)
                if status == "alert":
                    self.on_event({"type": "glass", "confidence": round(prob, 4),
                                   "t": time.strftime("%H:%M:%S"), "ts": time.time()})
        finally:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass

    def _emit_status(self, status, prob, rms, extra=None):
        ev = {"type": "audio", "status": status, "prob": round(prob, 4),
              "rms": round(rms, 5), "t": time.strftime("%H:%M:%S")}
        if extra:
            ev.update(extra)
        self.latest = ev
        self.on_event(ev)

    def _detect_clap(self, mono):
        if not self.clap_enabled:
            return
        peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
        now = time.time()
        if peak > self.clap_peak and (now - self._last_onset) > self.clap_refractory:
            self._last_onset = now
            self._clap_times.append(now)
            self._clap_times = [t for t in self._clap_times if now - t < 1.5]
            if len(self._clap_times) >= 2:
                gap = self._clap_times[-1] - self._clap_times[-2]
                if self.clap_gap[0] <= gap <= self.clap_gap[1]:
                    self._clap_times = []
                    self.on_event({"type": "clap", "count": 2, "gap": round(gap, 3),
                                   "t": time.strftime("%H:%M:%S"), "ts": time.time()})

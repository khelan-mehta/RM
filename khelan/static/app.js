const $ = s => document.querySelector(s);
const api = (p, opt) => fetch(p, opt).then(r => {
  if (r.status === 401) { location.href = "/login"; throw new Error("unauthorized"); }
  return r.json();
});
const ICON = {light:"💡", fan:"🌀", ac:"❄️", plug:"🔌", sensor:"📡"};
const EV_ICON = {glass:"🔨", security:"🚨", clap:"👏", device:"⚙️", voice:"🗣️",
                 alert_mode:"🛡️", };

let STATE = {devices:[], rooms:[], settings:{}, view:"rooms"};
let threshold = 0.8;

// ---------------- initial load ----------------
async function load(){
  const s = await api("/api/state");
  STATE.devices = s.devices; STATE.rooms = s.rooms; STATE.settings = s.settings;
  threshold = parseFloat(s.settings.glass_threshold || "0.8");
  $("#micName").textContent = "🎙 " + (s.mic?.name || "—");
  $("#thr").value = threshold; $("#thrVal").textContent = threshold.toFixed(2);
  $("#thrMark").style.left = (threshold*100)+"%";
  setAlert(s.settings.alert_mode === "on");
  renderAll();
  renderEvents(s.events);
  if (s.audio) applyAudio(s.audio);
}

function renderAll(){ renderDevices(); renderFloor(); populateSelects(); }

// ---------------- devices (rooms view) ----------------
function renderDevices(){
  const root = $("#roomsView"); root.innerHTML = "";
  const byRoom = {};
  STATE.rooms.forEach(r => byRoom[r.id] = []);
  STATE.devices.forEach(d => (byRoom[d.room_id] = byRoom[d.room_id] || []).push(d));
  STATE.rooms.forEach(room => {
    const list = byRoom[room.id] || [];
    const sec = document.createElement("div"); sec.className = "room";
    const on = list.filter(d => d.state.on).length;
    sec.innerHTML = `<div class="room-title"><span>${room.name}</span>
      <small>${on}/${list.length} on</small></div>`;
    const grid = document.createElement("div"); grid.className = "devgrid";
    if (!list.length) grid.innerHTML = `<p class="hint">No devices yet.</p>`;
    list.forEach(d => grid.appendChild(deviceCard(d)));
    sec.appendChild(grid); root.appendChild(sec);
  });
}

function deviceCard(d){
  const el = document.createElement("div");
  el.className = "dev" + (d.state.on ? " on" : "");
  el.dataset.id = d.id;
  let ctrl = "";
  if (d.state.on && d.type === "light")
    ctrl = sliderHtml("brightness", d.state.brightness ?? 100, 0, 100, "%");
  else if (d.state.on && d.type === "fan")
    ctrl = sliderHtml("speed", d.state.speed ?? 1, 0, 3, "");
  else if (d.state.on && d.type === "ac")
    ctrl = sliderHtml("temperature", d.state.temperature ?? 24, 16, 30, "°C");
  const hw = d.kind === "esp32";
  const badge = hw ? `<span class="badge ${d.online?"on":"off"}" title="${d.online?"online":"offline"}">ESP32 ${d.online?"●":"○"}</span>` : "";
  el.innerHTML = `
    <button class="dev-del" title="Remove">✕</button>
    <div class="dev-top"><span class="dev-ic">${ICON[d.type]||"▫"}</span>
      <button class="toggle ${d.state.on?"on":""}"></button></div>
    <h4>${d.name} ${badge}</h4>
    <div class="room-sub">${d.room_name||""}</div>
    <div class="state">${d.status||""}</div>
    <div class="dev-ctrl">${ctrl}</div>`;
  el.querySelector(".toggle").onclick = () => action(d.id, "toggle");
  el.querySelector(".dev-del").onclick = () => removeDevice(d.id);
  const rng = el.querySelector("input[type=range]");
  if (rng) rng.oninput = e => { clearTimeout(rng._t); rng._t = setTimeout(
      () => action(d.id, rng.dataset.field, +e.target.value), 120); };
  return el;
}
function sliderHtml(field, val, lo, hi, unit){
  return `<label>${field}<span>${val}${unit}</span></label>
    <input type="range" data-field="${field}" min="${lo}" max="${hi}" value="${val}">`;
}

async function action(id, action, value){
  const dev = await api(`/api/devices/${id}/action`, {method:"POST",
    headers:{"Content-Type":"application/json"}, body:JSON.stringify({action, value})});
  upsertDevice(dev);
}
async function removeDevice(id){
  await fetch(`/api/devices/${id}`, {method:"DELETE"});
  STATE.devices = STATE.devices.filter(d => d.id !== id); renderAll();
}
function upsertDevice(dev){
  const i = STATE.devices.findIndex(d => d.id === dev.id);
  if (i >= 0) STATE.devices[i] = {...STATE.devices[i], ...dev};
  else STATE.devices.push(dev);
  renderAll();
}

// ---------------- floor plan ----------------
function renderFloor(){
  const c = $("#floorCanvas"); if (!c) return; c.innerHTML = "";
  // rooms as zones (simple auto-grid if no coords set)
  STATE.rooms.forEach((r, i) => {
    const z = document.createElement("div"); z.className = "froom";
    const col = i % 3, row = Math.floor(i/3);
    z.style.left = (4 + col*32) + "%"; z.style.top = (6 + row*48) + "%";
    z.style.width = "29%"; z.style.height = "42%";
    z.innerHTML = `<span>${r.name}</span>`;
    c.appendChild(z);
  });
  STATE.devices.forEach(d => {
    const roomIdx = STATE.rooms.findIndex(r => r.id === d.room_id);
    const col = roomIdx<0?0:roomIdx % 3, row = roomIdx<0?0:Math.floor(roomIdx/3);
    const baseL = 4 + col*32, baseT = 6 + row*48;
    const x = d.x ?? 50, y = d.y ?? 50;
    const el = document.createElement("div");
    el.className = "fdev" + (d.state.on?" on":"");
    el.style.left = (baseL + (x/100)*25) + "%";
    el.style.top = (baseT + (y/100)*36) + "%";
    el.innerHTML = `${ICON[d.type]||"▫"}<small>${d.name}</small>`;
    el.dataset.id = d.id;
    makeDraggable(el, d, c);
    c.appendChild(el);
  });
}
function makeDraggable(el, d, canvas){
  let moved = false, sx, sy;
  el.addEventListener("pointerdown", e => {
    moved = false; sx = e.clientX; sy = e.clientY; el.setPointerCapture(e.pointerId);
    const move = ev => {
      if (Math.abs(ev.clientX-sx)+Math.abs(ev.clientY-sy) > 4) moved = true;
      const rect = canvas.getBoundingClientRect();
      el.style.left = ((ev.clientX-rect.left)/rect.width*100)+"%";
      el.style.top = ((ev.clientY-rect.top)/rect.height*100)+"%";
    };
    const up = ev => {
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      if (!moved){ action(d.id, "toggle"); return; }
      const rect = canvas.getBoundingClientRect();
      // store relative-to-room 0..100 (approx; room zone is 25%x36% of canvas)
      const roomIdx = STATE.rooms.findIndex(r => r.id === d.room_id);
      const col = roomIdx<0?0:roomIdx%3, row = roomIdx<0?0:Math.floor(roomIdx/3);
      const baseL = 4+col*32, baseT = 6+row*48;
      const px = ((ev.clientX-rect.left)/rect.width*100 - baseL)/25*100;
      const py = ((ev.clientY-rect.top)/rect.height*100 - baseT)/36*100;
      const nx = Math.max(0,Math.min(100,px)), ny = Math.max(0,Math.min(100,py));
      d.x = nx; d.y = ny;
      fetch(`/api/devices/${d.id}`, {method:"PATCH",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({x:nx, y:ny})});
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  });
}

// ---------------- selects / dialogs ----------------
function populateSelects(){
  const cd = $("#clapDevice"), cur = STATE.settings.clap_action_device || "";
  cd.innerHTML = `<option value="">— none —</option>` +
    STATE.devices.map(d => `<option value="${d.id}">${d.name} (${d.room_name||"—"})</option>`).join("");
  cd.value = cur;
  const nr = $("#ndRoom");
  nr.innerHTML = STATE.rooms.map(r => `<option value="${r.id}">${r.name}</option>`).join("");
}
$("#clapDevice").onchange = e => saveSetting("clap_action_device", e.target.value);

$("#addDeviceBtn").onclick = () => $("#deviceDialog").classList.remove("hidden");
document.querySelectorAll("[data-close]").forEach(b => b.onclick = () => $("#deviceDialog").classList.add("hidden"));
$("#ndSave").onclick = async () => {
  const kind = $("#ndKind").value;
  const body = {name:$("#ndName").value.trim()||"New Device", type:$("#ndType").value,
    room_id:+$("#ndRoom").value, kind};
  const dev = await api("/api/devices", {method:"POST",
    headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  upsertDevice(dev); $("#deviceDialog").classList.add("hidden"); $("#ndName").value="";
  if (kind === "esp32" && dev.token){
    const proto = location.protocol === "https:" ? "wss" : "ws";
    $("#espUrl").value = `${proto}://${location.host}/esp32?token=${dev.token}`;
    $("#espToken").value = dev.token;
    $("#espDialog").classList.remove("hidden");
  }
};
$("#espClose").onclick = () => $("#espDialog").classList.add("hidden");
$("#addRoomBtn").onclick = async () => {
  const name = prompt("Room name?"); if(!name) return;
  const room = await api("/api/rooms",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({name})});
  STATE.rooms.push(room); renderAll();
};

// tabs
document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
  t.classList.add("active");
  const v = t.dataset.view;
  $("#roomsView").classList.toggle("hidden", v!=="rooms");
  $("#floorView").classList.toggle("hidden", v!=="floor");
  if (v==="floor") renderFloor();
});

// ---------------- settings / alert ----------------
async function saveSetting(key, value){
  await fetch("/api/settings",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({key, value})});
  STATE.settings[key] = value;
}
$("#thr").oninput = e => {
  threshold = parseFloat(e.target.value);
  $("#thrVal").textContent = threshold.toFixed(2);
  $("#thrMark").style.left = (threshold*100)+"%";
  clearTimeout($("#thr")._t);
  $("#thr")._t = setTimeout(() => saveSetting("glass_threshold", threshold.toFixed(2)), 150);
};
$("#alertToggle").onchange = async e => {
  const r = await api("/api/alert_mode",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({on:e.target.checked})});
  setAlert(r.alert_mode === "on");
};
function setAlert(on){
  $("#alertToggle").checked = on;
  const p = $("#alertState");
  p.textContent = on ? "ARMED" : "DISARMED";
  p.className = "pill " + (on ? "s-armed" : "s-silence");
}

// ---------------- audio / websocket ----------------
function applyAudio(a){
  const prob = a.prob || 0;
  $("#probBig").textContent = (prob*100).toFixed(1)+"%";
  $("#gaugeFill").style.width = (prob*100)+"%";
  $("#rmsVal").textContent = (a.rms||0).toFixed(4);
  const p = $("#statusPill");
  const st = a.status;
  const label = st==="alert"?"GLASS":st==="silence"?"SILENCE":
    st==="listening"?"LISTENING":st==="loading-model"?"LOADING…":
    st==="no-mic"||st==="mic-error"?"NO MIC":st?.toUpperCase()||"—";
  p.textContent = label;
  p.className = "pill " + (st==="alert"?"s-alert":st==="listening"?"s-listening":"s-silence");
}
function renderEvents(list){
  const ul = $("#events");
  if (!list || !list.length){ ul.innerHTML = `<li class="empty">No events yet.</li>`; return; }
  ul.innerHTML = list.map(e => {
    const d = new Date(e.ts*1000);
    const tm = d.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"});
    return `<li class="ev-${e.type}"><span class="ev-ic">${EV_ICON[e.type]||"•"}</span>
      <span class="ev-msg">${e.message||e.type}</span><span class="ev-time">${tm}</span></li>`;
  }).join("");
}
function prependEvent(e){
  const ul = $("#events"); const empty = ul.querySelector(".empty"); if (empty) ul.innerHTML = "";
  const d = new Date((e.ts||Date.now()/1000)*1000);
  const tm = d.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"});
  const li = document.createElement("li"); li.className = "ev-"+e.type;
  li.innerHTML = `<span class="ev-ic">${EV_ICON[e.type]||"•"}</span>
    <span class="ev-msg">${e.message||e.type}</span><span class="ev-time">${tm}</span>`;
  ul.prepend(li); while (ul.children.length > 100) ul.removeChild(ul.lastChild);
}
function toast(msg){
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3000);
}
function beep(){ try{
  const ac = new (window.AudioContext||window.webkitAudioContext)();
  const o = ac.createOscillator(), g = ac.createGain();
  o.type="square"; o.frequency.value=880; o.connect(g); g.connect(ac.destination);
  g.gain.setValueAtTime(.001,ac.currentTime); g.gain.exponentialRampToValueAtTime(.25,ac.currentTime+.02);
  g.gain.exponentialRampToValueAtTime(.001,ac.currentTime+.4); o.start(); o.stop(ac.currentTime+.42);
}catch(e){} }
function notify(title, body){
  if ("Notification" in window && Notification.permission === "granted")
    new Notification(title, {body});
}

function connect(){
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => { $("#dot").className = "dot live"; $("#connText").textContent = "live"; };
  ws.onclose = (e) => {
    if (e.code === 1008) { location.href = "/login"; return; }
    $("#dot").className = "dot"; $("#connText").textContent = "reconnecting…";
    setTimeout(connect, 1500);
  };
  ws.onmessage = ev => handle(JSON.parse(ev.data));
  window._ws = ws;
}
function handle(m){
  switch(m.kind){
    case "audio": applyAudio(m); break;
    case "device_update": case "device_added": upsertDevice(m.device);
      if (m.event) prependEvent(m.event); break;
    case "device_removed":
      STATE.devices = STATE.devices.filter(d => d.id !== m.id); renderAll(); break;
    case "detection":
      prependEvent(m.event);
      if (m.notify){ beep(); toast("🚨 GLASS BREAK DETECTED"); notify("Khelan Alert","Glass break detected"); }
      break;
    case "security":
      prependEvent(m.event); beep(); toast("🚨 SECURITY ALERT"); notify("Khelan Security","Glass break while armed!"); break;
    case "clap":
      prependEvent(m.event); if (m.device){ upsertDevice(m.device); } toast("👏 Double-clap"); break;
    case "alert_mode": setAlert(m.on); if (m.event) prependEvent(m.event); break;
    case "room_added": case "rooms_changed": load(); break;
    case "settings":
      STATE.settings[m.key] = m.value;
      if (m.key==="glass_threshold"){ threshold=parseFloat(m.value);
        $("#thr").value=threshold; $("#thrVal").textContent=threshold.toFixed(2);
        $("#thrMark").style.left=(threshold*100)+"%"; }
      break;
  }
}

// ---------------- voice (Web Speech API) ----------------
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
function speak(t){ try{ speechSynthesis.speak(new SpeechSynthesisUtterance(t)); }catch(e){} }
async function runCommand(text){
  $("#voiceReply").textContent = `“${text}” …`;
  const r = await api("/api/voice",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({text})});
  $("#voiceReply").textContent = r.reply;
  speak(r.reply);
  if (r.device) upsertDevice(r.device);
}
$("#voiceSend").onclick = () => { const t=$("#voiceText").value.trim(); if(t) runCommand(t); };
$("#voiceText").addEventListener("keydown", e => { if(e.key==="Enter"){ const t=e.target.value.trim(); if(t) runCommand(t);} });

let recog=null, listening=false;
$("#listenBtn").onclick = () => {
  if (!SR){ $("#voiceReply").textContent = "Speech recognition not supported in this browser (try Chrome)."; return; }
  if (listening){ recog && recog.stop(); return; }
  recog = new SR(); recog.lang="en-US"; recog.interimResults=false; recog.maxAlternatives=1;
  recog.onstart = () => { listening=true; $("#listenBtn").classList.add("listening"); $("#listenBtn").textContent="🎤 Listening…"; };
  recog.onend = () => { listening=false; $("#listenBtn").classList.remove("listening"); $("#listenBtn").textContent="🎤 Hold / Tap to speak"; };
  recog.onerror = e => { $("#voiceReply").textContent = "Mic error: "+e.error; };
  recog.onresult = e => { const t=e.results[0][0].transcript; $("#voiceText").value=t; runCommand(t); };
  recog.start();
};

// wake-word continuous mode
let wake=null;
$("#wakeChk").onchange = e => {
  if (!SR){ e.target.checked=false; toast("Speech not supported"); return; }
  if (e.target.checked){
    wake = new SR(); wake.continuous=true; wake.interimResults=false; wake.lang="en-US";
    wake.onresult = ev => {
      const t = ev.results[ev.results.length-1][0].transcript.toLowerCase();
      if (t.includes("khelan")){ const cmd = t.split("khelan").pop().trim(); if (cmd) runCommand(cmd); }
    };
    wake.onend = () => { if ($("#wakeChk").checked) try{wake.start()}catch(e){} };
    try{ wake.start(); toast("Say “Hey Khelan …”"); }catch(e){}
  } else { wake && wake.stop(); wake=null; }
};

// auth controls
$("#logoutBtn").onclick = async () => {
  await fetch("/api/logout", {method:"POST"}); location.href = "/login";
};
$("#pwBtn").onclick = async () => {
  const np = prompt("New password (min 4 chars):"); if (!np) return;
  const r = await fetch("/api/password", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({new_password: np})});
  toast(r.ok ? "Password changed" : "Could not change password");
};

// notifications permission on first interaction
document.body.addEventListener("click", () => {
  if ("Notification" in window && Notification.permission === "default") Notification.requestPermission();
}, {once:true});

load(); connect();

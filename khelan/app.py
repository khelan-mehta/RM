"""Khelan — AI smart-home automation backend (FastAPI).

One integrated backend serving a responsive web dashboard, device control
(virtual now, ESP32-ready later), rooms + floor plan, sound recognition and
double-clap automation (via the audio engine), Alert Mode, voice-command
execution, and real-time updates over WebSocket.
"""
import asyncio
import json
import os
import queue
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db import DB
from devices import apply_action, summarize, parse_intent
from audio_engine import AudioEngine
import auth

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

import secrets

db: DB = None
engine: AudioEngine = None
OUT = queue.Queue()                 # events (from any thread) → dashboard websocket broadcast
ESP = {}                            # device_id → connected ESP32 websocket (hardware)
ESP_OUT = queue.Queue()             # (device_id, command dict) → send to that board


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def enqueue(msg):
    OUT.put(msg)


def devices_with_rooms():
    rooms = {r["id"]: r["name"] for r in db.list_rooms()}
    out = []
    for d in db.list_devices():
        d["room_name"] = rooms.get(d["room_id"], "")
        d["status"] = summarize(d)
        d.pop("token", None)                       # never expose the device token
        d["online"] = (d["kind"] != "esp32") or (d["id"] in ESP)
        out.append(d)
    return out


def _public(dev):
    dev = dict(dev); dev.pop("token", None)
    dev["online"] = (dev["kind"] != "esp32") or (dev["id"] in ESP)
    dev["status"] = summarize(dev)
    return dev


def execute_action(device_id, action, value=None, source="api"):
    dev = db.get_device(device_id)
    if not dev:
        return None
    new_state = apply_action(dev, action, value)
    dev = db.set_device_state(device_id, new_state)
    pub = _public(dev)
    # Real hardware: forward the desired state to the connected board.
    if dev["kind"] == "esp32":
        ESP_OUT.put((device_id, {"cmd": "set", "state": new_state}))
    rec = db.add_event("device", source=source, message=f"{dev['name']} → {summarize(dev)}",
                       device_id=device_id)
    enqueue({"kind": "device_update", "device": pub, "event": rec})
    return pub


async def esp32_sender():
    """Deliver queued commands to connected ESP32 boards (runs on the loop)."""
    while True:
        try:
            while True:
                device_id, cmd = ESP_OUT.get_nowait()
                ws = ESP.get(device_id)
                if ws is not None:
                    try:
                        await ws.send_text(json.dumps(cmd))
                    except Exception:
                        pass
        except queue.Empty:
            pass
        await asyncio.sleep(0.05)


# ---- audio engine event handler (runs on the audio thread) ----
def on_audio_event(ev):
    t = ev["type"]
    if t == "audio":
        enqueue({"kind": "audio", **ev})
    elif t == "glass":
        armed = db.get_setting("alert_mode", "off") == "on"
        conf = ev["confidence"]
        rec = db.add_event("glass", source="sound",
                           message=f"Glass break detected ({conf*100:.0f}%)", confidence=conf)
        enqueue({"kind": "detection", "sound": "glass", "event": rec,
                 "armed": armed, "notify": armed})
        if armed:
            srec = db.add_event("security", source="alert_mode",
                                message="SECURITY ALERT · glass break while armed", confidence=conf)
            enqueue({"kind": "security", "event": srec, "notify": True})
    elif t == "clap":
        rec = db.add_event("clap", source="mic", message="Double-clap detected")
        msg = {"kind": "clap", "event": rec}
        dev_id = db.get_setting("clap_action_device", "")
        if dev_id:
            try:
                dev = execute_action(int(dev_id), "toggle", source="clap")
                if dev:
                    msg["device"] = dev
            except (ValueError, TypeError):
                pass
        enqueue(msg)


# ---------------------------------------------------------------------------
# websocket connection manager
# ---------------------------------------------------------------------------
class Manager:
    def __init__(self):
        self.active = set()

    async def connect(self, ws):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws):
        self.active.discard(ws)

    async def broadcast(self, msg):
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = Manager()


async def broadcaster():
    while True:
        try:
            while True:
                msg = OUT.get_nowait()
                await manager.broadcast(msg)
        except queue.Empty:
            pass
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, engine
    db = DB()
    new_pw = auth.ensure_admin(db)
    if new_pw:
        print(f"\n*** Khelan login created → username: admin  password: {new_pw}")
        print("    (also saved to khelan/ADMIN_LOGIN.txt) ***\n")
    try:
        threshold = float(db.get_setting("glass_threshold", "0.8"))
    except (TypeError, ValueError):
        threshold = 0.8
    device_spec = os.environ.get("KHELAN_MIC", db.get_setting("mic_device", "default"))
    engine = AudioEngine(on_audio_event, device=device_spec, threshold=threshold)
    engine.start()
    task = asyncio.create_task(broadcaster())
    esp_task = asyncio.create_task(esp32_sender())
    yield
    task.cancel(); esp_task.cancel()
    if engine:
        engine.stop()


app = FastAPI(title="Khelan", lifespan=lifespan)

# Paths reachable without a session (login page + its assets + the login API).
_PUBLIC = {"/login", "/api/login", "/styles.css", "/favicon.ico"}


@app.middleware("http")
async def require_auth(request: Request, call_next):
    path = request.url.path
    if path in _PUBLIC:
        return await call_next(request)
    user = auth.verify_token(db, request.cookies.get(auth.COOKIE)) if db else None
    if not user:
        if path.startswith("/api/"):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return RedirectResponse("/login")
    return await call_next(request)


class LoginIn(BaseModel):
    username: str
    password: str


class PasswordIn(BaseModel):
    new_password: str


@app.post("/api/login")
def login(body: LoginIn, request: Request):
    if auth.check_login(db, body.username, body.password):
        token = auth.make_token(db, body.username)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(auth.COOKIE, token, httponly=True, samesite="lax",
                        max_age=auth.SESSION_TTL, secure=(request.url.scheme == "https"))
        return resp
    return JSONResponse({"ok": False, "detail": "invalid credentials"}, status_code=401)


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE)
    return resp


@app.post("/api/password")
def change_password(body: PasswordIn):
    if len(body.new_password) < 4:
        raise HTTPException(400, "password too short")
    auth.set_password(db, body.new_password)
    return {"ok": True}


@app.get("/login")
def login_page():
    return FileResponse(os.path.join(STATIC, "login.html"))


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------
class DeviceIn(BaseModel):
    name: str
    type: str
    room_id: int | None = None
    x: float | None = None
    y: float | None = None
    kind: str | None = "virtual"


class DevicePatch(BaseModel):
    name: str | None = None
    room_id: int | None = None
    x: float | None = None
    y: float | None = None


class ActionIn(BaseModel):
    action: str
    value: object | None = None


class RoomIn(BaseModel):
    name: str


class RoomPatch(BaseModel):
    name: str | None = None
    x: float | None = None
    y: float | None = None
    w: float | None = None
    h: float | None = None
    ord: int | None = None


class VoiceIn(BaseModel):
    text: str


class SettingIn(BaseModel):
    key: str
    value: object


class AlertIn(BaseModel):
    on: bool


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/state")
def get_state():
    return {
        "rooms": db.list_rooms(),
        "devices": devices_with_rooms(),
        "events": db.list_events(50),
        "audio": engine.latest if engine else {},
        "mic": {"name": engine.device_name if engine else "—",
                "available": engine.available if engine else False},
        "settings": {
            "alert_mode": db.get_setting("alert_mode", "off"),
            "clap_action_device": db.get_setting("clap_action_device", ""),
            "glass_threshold": db.get_setting("glass_threshold", "0.8"),
        },
    }


@app.get("/api/devices")
def api_devices():
    return devices_with_rooms()


@app.post("/api/devices")
def api_add_device(d: DeviceIn):
    kind = d.kind or "virtual"
    did = db.add_device(d.name, d.type, d.room_id, d.x or 50, d.y or 50, kind=kind)
    token = None
    if kind == "esp32":
        token = secrets.token_urlsafe(18)
        db.set_device_token(did, token)
    pub = _public(db.get_device(did))
    enqueue({"kind": "device_added", "device": pub})
    # token is returned ONCE to the creator so the board can be flashed with it
    return {**pub, "token": token}


@app.post("/api/devices/{device_id}/token")
def api_reissue_token(device_id: int):
    dev = db.get_device(device_id)
    if not dev:
        raise HTTPException(404, "device not found")
    token = secrets.token_urlsafe(18)
    db.set_device_token(device_id, token)
    return {"device_id": device_id, "token": token}


@app.patch("/api/devices/{device_id}")
def api_patch_device(device_id: int, p: DevicePatch):
    dev = db.update_device(device_id, **p.model_dump(exclude_none=True))
    if not dev:
        raise HTTPException(404, "device not found")
    dev["status"] = summarize(dev)
    enqueue({"kind": "device_update", "device": dev})
    return dev


@app.post("/api/devices/{device_id}/action")
def api_action(device_id: int, a: ActionIn):
    dev = execute_action(device_id, a.action, a.value, source="api")
    if not dev:
        raise HTTPException(404, "device not found")
    return dev


@app.delete("/api/devices/{device_id}")
def api_delete_device(device_id: int):
    db.delete_device(device_id)
    enqueue({"kind": "device_removed", "id": device_id})
    return {"ok": True}


@app.get("/api/rooms")
def api_rooms():
    return db.list_rooms()


@app.post("/api/rooms")
def api_add_room(r: RoomIn):
    rid = db.add_room(r.name, ord=len(db.list_rooms()) + 1)
    room = next((x for x in db.list_rooms() if x["id"] == rid), None)
    enqueue({"kind": "room_added", "room": room})
    return room


@app.patch("/api/rooms/{room_id}")
def api_patch_room(room_id: int, p: RoomPatch):
    db.update_room(room_id, **p.model_dump(exclude_none=True))
    enqueue({"kind": "rooms_changed"})
    return {"ok": True}


@app.delete("/api/rooms/{room_id}")
def api_delete_room(room_id: int):
    db.delete_room(room_id)
    enqueue({"kind": "rooms_changed"})
    return {"ok": True}


@app.get("/api/events")
def api_events(limit: int = 100):
    return db.list_events(limit)


@app.post("/api/settings")
def api_settings(s: SettingIn):
    db.set_setting(s.key, s.value)
    if s.key == "glass_threshold" and engine:
        try:
            engine.threshold = float(s.value)
        except (TypeError, ValueError):
            pass
    enqueue({"kind": "settings", "key": s.key, "value": str(s.value)})
    return {"ok": True}


@app.post("/api/alert_mode")
def api_alert_mode(a: AlertIn):
    db.set_setting("alert_mode", "on" if a.on else "off")
    rec = db.add_event("alert_mode", source="user",
                       message=f"Alert Mode {'ARMED' if a.on else 'disarmed'}")
    enqueue({"kind": "alert_mode", "on": a.on, "event": rec})
    return {"alert_mode": "on" if a.on else "off"}


@app.post("/api/voice")
def api_voice(v: VoiceIn):
    intent = parse_intent(v.text, devices_with_rooms())
    if not intent:
        return {"ok": False, "reply": "Sorry, I didn't catch a device command.",
                "transcript": v.text}
    dev = execute_action(intent["device_id"], intent["action"], intent["value"], source="voice")
    db.add_event("voice", source="voice", message=f'"{v.text}" → {intent["reply"]}')
    return {"ok": True, "reply": intent["reply"], "device": dev, "transcript": v.text}


# ---------------------------------------------------------------------------
# websocket + static
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    if not auth.verify_token(db, websocket.cookies.get(auth.COOKIE)):
        await websocket.close(code=1008)   # policy violation
        return
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({"kind": "hello"}))
        while True:
            await websocket.receive_text()   # keep-alive; client doesn't need to send
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@app.websocket("/esp32")
async def esp32_ws(websocket: WebSocket):
    """Endpoint for real ESP32 boards. Auth is the per-device token (query param
    ?token=...), NOT the user session. The board receives {cmd:"set",state}
    commands and may send {type:"state"|"sound"|"sensor", ...} reports."""
    token = websocket.query_params.get("token")
    dev = db.device_by_token(token) if db else None
    if not dev:
        await websocket.close(code=1008)
        return
    device_id = dev["id"]
    await websocket.accept()
    ESP[device_id] = websocket
    # push current desired state so the board syncs on connect
    try:
        await websocket.send_text(json.dumps({"cmd": "set", "state": dev["state"]}))
    except Exception:
        pass
    enqueue({"kind": "device_update", "device": _public(db.get_device(device_id)),
             "event": db.add_event("device", source="esp32",
                                   message=f"{dev['name']} connected", device_id=device_id)})
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type")
            if mtype == "state" and isinstance(msg.get("state"), dict):
                db.set_device_state(device_id, msg["state"])
                enqueue({"kind": "device_update", "device": _public(db.get_device(device_id))})
            elif mtype == "sound" and msg.get("sound") == "glass":
                armed = db.get_setting("alert_mode", "off") == "on"
                conf = float(msg.get("confidence", 1.0))
                rec = db.add_event("glass", source="esp32",
                                   message=f"Glass break (board: {dev['name']}) {conf*100:.0f}%",
                                   confidence=conf, device_id=device_id)
                enqueue({"kind": "detection", "sound": "glass", "event": rec,
                         "armed": armed, "notify": armed})
            elif mtype == "sensor":
                rec = db.add_event("sensor", source="esp32",
                                   message=f"{dev['name']}: {msg.get('message','sensor event')}",
                                   device_id=device_id)
                enqueue({"kind": "event", "event": rec})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ESP.get(device_id) is websocket:
            del ESP[device_id]
        enqueue({"kind": "device_update", "device": _public(db.get_device(device_id)),
                 "event": db.add_event("device", source="esp32",
                                       message=f"{dev['name']} disconnected", device_id=device_id)})


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/", StaticFiles(directory=STATIC), name="static")

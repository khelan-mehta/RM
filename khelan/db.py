"""SQLite storage for Khelan: rooms, devices, settings, and event history.

Uses the Python standard-library sqlite3 (no extra dependency). One connection,
guarded by a lock, shared across the FastAPI request threads and the audio
thread (check_same_thread=False).
"""
import json
import os
import sqlite3
import threading
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "khelan.db")

_DEFAULT_STATE = {
    "light": {"on": False, "brightness": 100},
    "fan":   {"on": False, "speed": 1},
    "ac":    {"on": False, "temperature": 24, "mode": "cool"},
    "plug":  {"on": False},
    "sensor": {"on": True},
}


class DB:
    def __init__(self, path=DB_PATH):
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        if not self.list_rooms():
            self._seed()

    # ---------- schema ----------
    def _init_schema(self):
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rooms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    ord INTEGER DEFAULT 0,
                    x REAL DEFAULT 0, y REAL DEFAULT 0, w REAL DEFAULT 24, h REAL DEFAULT 18
                );
                CREATE TABLE IF NOT EXISTS devices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    room_id INTEGER,
                    state TEXT NOT NULL,
                    x REAL DEFAULT 50, y REAL DEFAULT 50,
                    kind TEXT DEFAULT 'virtual',
                    FOREIGN KEY(room_id) REFERENCES rooms(id) ON DELETE SET NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    type TEXT NOT NULL,
                    source TEXT,
                    message TEXT,
                    confidence REAL,
                    device_id INTEGER
                );
                """
            )
            # migrations (safe to run repeatedly)
            cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(devices)")}
            if "token" not in cols:
                self.conn.execute("ALTER TABLE devices ADD COLUMN token TEXT")
            self.conn.commit()

    def set_device_token(self, device_id, token):
        with self.lock:
            self.conn.execute("UPDATE devices SET token=? WHERE id=?", (token, device_id))
            self.conn.commit()

    def device_by_token(self, token):
        if not token:
            return None
        with self.lock:
            r = self.conn.execute("SELECT * FROM devices WHERE token=?", (token,)).fetchone()
            if not r:
                return None
            d = dict(r); d["state"] = json.loads(d["state"]); return d

    def _seed(self):
        rooms = [("Living Room", 1), ("Bedroom", 2), ("Kitchen", 3)]
        ids = {}
        for name, o in rooms:
            ids[name] = self.add_room(name, o)
        seed_devices = [
            ("Ceiling Light", "light", "Living Room", 30, 30),
            ("TV Plug", "plug", "Living Room", 70, 60),
            ("Bedroom AC", "ac", "Bedroom", 40, 40),
            ("Bedroom Light", "light", "Bedroom", 70, 30),
            ("Bedroom Fan", "fan", "Bedroom", 30, 70),
            ("Kitchen Light", "light", "Kitchen", 50, 40),
        ]
        for name, typ, room, x, y in seed_devices:
            self.add_device(name, typ, ids[room], x, y)
        self.set_setting("alert_mode", "off")
        self.set_setting("clap_action_device", "")

    # ---------- rooms ----------
    def list_rooms(self):
        with self.lock:
            return [dict(r) for r in self.conn.execute("SELECT * FROM rooms ORDER BY ord, id")]

    def add_room(self, name, ord=0, x=0, y=0, w=24, h=18):
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO rooms(name, ord, x, y, w, h) VALUES (?,?,?,?,?,?)",
                (name, ord, x, y, w, h))
            self.conn.commit()
            return cur.lastrowid

    def update_room(self, room_id, **fields):
        allowed = {"name", "ord", "x", "y", "w", "h"}
        sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not sets:
            return
        with self.lock:
            cols = ", ".join(f"{k}=?" for k in sets)
            self.conn.execute(f"UPDATE rooms SET {cols} WHERE id=?", (*sets.values(), room_id))
            self.conn.commit()

    def delete_room(self, room_id):
        with self.lock:
            self.conn.execute("DELETE FROM rooms WHERE id=?", (room_id,))
            self.conn.commit()

    # ---------- devices ----------
    def list_devices(self):
        with self.lock:
            out = []
            for r in self.conn.execute("SELECT * FROM devices ORDER BY id"):
                d = dict(r)
                d["state"] = json.loads(d["state"])
                out.append(d)
            return out

    def get_device(self, device_id):
        with self.lock:
            r = self.conn.execute("SELECT * FROM devices WHERE id=?", (device_id,)).fetchone()
            if not r:
                return None
            d = dict(r); d["state"] = json.loads(d["state"])
            return d

    def add_device(self, name, type, room_id=None, x=50, y=50, kind="virtual", state=None):
        state = state or dict(_DEFAULT_STATE.get(type, {"on": False}))
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO devices(name, type, room_id, state, x, y, kind) VALUES (?,?,?,?,?,?,?)",
                (name, type, room_id, json.dumps(state), x, y, kind))
            self.conn.commit()
            return cur.lastrowid

    def update_device(self, device_id, **fields):
        allowed = {"name", "type", "room_id", "x", "y", "kind"}
        sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if "state" in fields and fields["state"] is not None:
            sets["state"] = json.dumps(fields["state"])
        if not sets:
            return self.get_device(device_id)
        with self.lock:
            cols = ", ".join(f"{k}=?" for k in sets)
            self.conn.execute(f"UPDATE devices SET {cols} WHERE id=?", (*sets.values(), device_id))
            self.conn.commit()
        return self.get_device(device_id)

    def set_device_state(self, device_id, state):
        with self.lock:
            self.conn.execute("UPDATE devices SET state=? WHERE id=?", (json.dumps(state), device_id))
            self.conn.commit()
        return self.get_device(device_id)

    def delete_device(self, device_id):
        with self.lock:
            self.conn.execute("DELETE FROM devices WHERE id=?", (device_id,))
            self.conn.commit()

    # ---------- settings ----------
    def get_setting(self, key, default=None):
        with self.lock:
            r = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return r["value"] if r else default

    def set_setting(self, key, value):
        with self.lock:
            self.conn.execute(
                "INSERT INTO settings(key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
            self.conn.commit()

    # ---------- events ----------
    def add_event(self, type, source=None, message=None, confidence=None, device_id=None, ts=None):
        ts = ts or time.time()
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO events(ts, type, source, message, confidence, device_id) VALUES (?,?,?,?,?,?)",
                (ts, type, source, message, confidence, device_id))
            self.conn.commit()
            return {"id": cur.lastrowid, "ts": ts, "type": type, "source": source,
                    "message": message, "confidence": confidence, "device_id": device_id}

    def list_events(self, limit=100):
        with self.lock:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))]

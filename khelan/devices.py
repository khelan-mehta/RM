"""Virtual device actions and a lightweight voice-intent parser.

Virtual devices are pure state in the DB. The same action functions are what an
ESP32 adapter would later forward to real hardware, so the control path is
identical for simulated and real devices.
"""
import re

CLAMP = {
    "brightness": (0, 100),
    "speed": (0, 3),
    "temperature": (16, 30),
}


def apply_action(device, action, value=None):
    """Return a new state dict for `device` after applying `action`.

    Supported actions: on, off, toggle, set (with field via value dict),
    brightness/speed/temperature (value = number), mode (value = str).
    """
    state = dict(device["state"])
    typ = device["type"]

    if action == "on":
        state["on"] = True
    elif action == "off":
        state["on"] = False
    elif action == "toggle":
        state["on"] = not state.get("on", False)
    elif action in ("brightness", "speed", "temperature"):
        lo, hi = CLAMP[action]
        try:
            v = int(round(float(value)))
        except (TypeError, ValueError):
            v = state.get(action, lo)
        state[action] = max(lo, min(hi, v))
        state["on"] = True
    elif action == "mode" and value:
        state["mode"] = str(value)
        state["on"] = True
    elif action == "set" and isinstance(value, dict):
        for k, v in value.items():
            if k in CLAMP:
                lo, hi = CLAMP[k]
                try:
                    v = max(lo, min(hi, int(round(float(v)))))
                except (TypeError, ValueError):
                    continue
            state[k] = v
    return state


def summarize(device):
    s = device["state"]
    if not s.get("on", False):
        return "off"
    t = device["type"]
    if t == "light":
        return f"on · {s.get('brightness', 100)}%"
    if t == "fan":
        return f"on · speed {s.get('speed', 1)}"
    if t == "ac":
        return f"on · {s.get('temperature', 24)}°C {s.get('mode', 'cool')}"
    return "on"


# ---------------------------------------------------------------------------
# Voice / text intent parsing  ("turn on the bedroom AC", "set bedroom ac to 20")
# ---------------------------------------------------------------------------
_ROOM_WORDS = ["living room", "bedroom", "kitchen", "bathroom", "hall", "garage",
               "office", "balcony", "dining"]
_TYPE_WORDS = {"light": "light", "lights": "light", "lamp": "light",
               "fan": "fan", "ac": "ac", "a.c.": "ac", "air conditioner": "ac",
               "ac ": "ac", "plug": "plug", "socket": "plug", "tv": "plug"}


def parse_intent(text, devices):
    """Map a free-text command to (device_id, action, value, reply) using the
    current device list. Returns dict or None if nothing matched."""
    t = " " + text.lower().strip() + " "

    # action
    action, value = None, None
    if re.search(r"\b(turn on|switch on|power on|enable|start)\b", t) or re.search(r"\bon\b", t):
        action = "on"
    if re.search(r"\b(turn off|switch off|power off|disable|stop|shut)\b", t) or re.search(r"\boff\b", t):
        action = "off"
    if re.search(r"\btoggle\b", t):
        action = "toggle"
    m = re.search(r"(set|make|change).*?(\d{1,3})", t)
    if re.search(r"\b(temperature|degrees|°|cool to|set .*ac)\b", t) and re.search(r"\d", t):
        mm = re.search(r"(\d{1,2})", t)
        if mm:
            action, value = "temperature", int(mm.group(1))
    elif re.search(r"\b(brightness|dim|bright)\b", t) and re.search(r"\d", t):
        mm = re.search(r"(\d{1,3})", t)
        if mm:
            action, value = "brightness", int(mm.group(1))
    elif m and action is None:
        value = int(m.group(2))
        action = "set"

    # room
    room = next((r for r in _ROOM_WORDS if r in t), None)
    # type
    dev_type = None
    for w, typ in _TYPE_WORDS.items():
        if w.strip() and w in t:
            dev_type = typ
            break

    # match a device
    def score(d):
        s = 0
        dn = d["name"].lower()
        if dn in t:
            s += 5
        if dev_type and d["type"] == dev_type:
            s += 2
        if room and d.get("room_name", "").lower() == room:
            s += 2
        if room and room.split()[0] in dn:
            s += 1
        return s

    ranked = sorted(devices, key=score, reverse=True)
    best = ranked[0] if ranked else None
    if not best or score(best) == 0 or action is None:
        return None

    val = value
    if action in ("brightness", "temperature", "speed"):
        pass
    reply_target = best["name"]
    if action == "on":
        reply = f"Turning on {reply_target}."
    elif action == "off":
        reply = f"Turning off {reply_target}."
    elif action == "toggle":
        reply = f"Toggling {reply_target}."
    elif action == "temperature":
        reply = f"Setting {reply_target} to {val} degrees."
    elif action == "brightness":
        reply = f"Setting {reply_target} brightness to {val} percent."
    else:
        reply = f"Updating {reply_target}."
    return {"device_id": best["id"], "action": action, "value": val, "reply": reply}

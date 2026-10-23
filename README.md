# Khelan — AI Smart-Home Automation

An integrated, responsive **web application** that turns a home into an AI-assisted smart space:
voice control, double-clap automation, sound-based security (glass-break detection), a
configurable floor plan, virtual devices with a path to real **ESP32** hardware, secure
login + LAN remote access — all served by one Python backend. No separate mobile app.

Built on top of an acoustic **glass-break detection** CNN (mel-spectrogram → Keras model,
~96% test accuracy, 93/93 precision/recall).

---

## Features

- **Dashboard** — live device control, sound monitor, and event history over WebSocket.
- **Sound recognition** — reuses the glass-break CNN on a rolling 2 s mic window
  (native-rate capture resampled to 22.05 kHz; `glass P = 1 − sigmoid`).
- **Double-clap automation** — lightweight transient detector toggles a chosen device.
- **Alert Mode** — arm when leaving; glass-break/alarm sounds raise a security alert + notification.
- **Voice assistant** — browser Web Speech API ("Hey Khelan") → intent parser → device action → spoken reply.
- **Virtual devices + rooms + interactive floor plan** — drag-to-arrange, click-to-toggle, persisted in SQLite.
- **Secure access** — login (PBKDF2 + HMAC-signed sessions); HTTPS LAN mode for phones.
- **ESP32 integration** — real boards connect over an authenticated WebSocket and are driven
  through the **same** control path as virtual devices (reference firmware included).

## Architecture

```
Browser (dashboard, voice)  ──WS/REST──┐
ESP32 boards  ──authenticated WS──────┐│
                                      ▼▼
                FastAPI backend (khelan/)
   ┌───────────────┬───────────────┬──────────────┐
   │ audio_engine  │ devices /     │ auth + SQLite │
   │ (mic→CNN,     │ apply_action  │ (rooms,       │
   │  double-clap) │ (virtual/ESP) │  events)      │
   └───────────────┴───────────────┴──────────────┘
```

## Project structure

```
khelan/                 FastAPI app
  app.py                routes, WebSocket, ESP32 bridge
  audio_engine.py       mic capture, glass CNN, double-clap
  devices.py            device actions + voice intent parser
  db.py                 SQLite storage
  auth.py               login + signed-cookie sessions
  certs.py              self-signed cert for HTTPS LAN mode
  run.py                launcher (local / --remote)
  static/               dashboard + login UI (vanilla JS)
  firmware/khelan_esp32.ino   reference ESP32 firmware
src/                    glass-detection ML scripts (train/evaluate/predict)
models/glass_detector_v2.keras   trained model used at runtime
```

## Setup

Requires **Python 3.11** (TensorFlow 2.21 wheels).

```bash
python -m venv venv
# Windows:  venv\Scripts\activate      Linux/Mac:  source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Local (localhost only):
python khelan/run.py                     # http://127.0.0.1:8000

# Phone / LAN over HTTPS (self-signed, auth required):
python khelan/run.py --remote            # https://<your-lan-ip>:8443
```

Environment variables: `KHELAN_PORT`, `KHELAN_HOST`, `KHELAN_MIC` (index / name / `default`),
`KHELAN_PASSWORD` (sets the admin password on first run). If no password is set, a random one
is generated and written to `khelan/ADMIN_LOGIN.txt`.

## ESP32 hardware

In the dashboard: **+ Device → Kind: ESP32**. Copy the shown WebSocket URL + token into
`khelan/firmware/khelan_esp32.ino` (needs the `WebSockets` and `ArduinoJson` libraries),
flash the board, and it appears **online** and responds to the same controls as a virtual device.

## Notes

- The microphone is accessed only locally; no audio leaves the machine.
- Secrets (SQLite DB, tokens, TLS key, generated password) are git-ignored.
- GPU is not used on native Windows (TensorFlow ≥ 2.11); inference runs on CPU.

## Credits

Research Methodology project — Manthan Patel (23BEC140), Khelan Mehta (23BEC088),
Institute of Technology, Nirma University. Mentor: Dr. Shefali Waldekar.

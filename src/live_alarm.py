"""
Real-time glass-break alarm.

Continuously listens to the microphone using a rolling audio window, turns each
window into a mel-spectrogram (matching how the training images were generated),
runs it through the trained CNN, and raises an alert when a glass break is
detected.

This replaces the manual 3-step flow (record -> spectrogram -> predict) with a
single always-on loop. The model is loaded once, up front, instead of on every
prediction.

Run from the project root:
    python src/live_alarm.py
Stop with Ctrl+C.
"""

import argparse
import io
import os
import sys
import threading
import time

# Quiet TensorFlow's C++ logging before it is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Make emoji / unicode printable on a legacy Windows (cp1252) console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")  # headless rendering, no GUI window
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# ===============================
# Defaults (match training)
# ===============================

FS = 22050            # sample rate used everywhere in this project
N_MELS = 128
IMG_SIZE = (128, 128)
# Resolve the default model relative to the project root (parent of src/) so the
# script works no matter which directory it is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "glass_detector_v2.keras")


def make_spectrogram_array(y, sr):
    """Turn a 1-D audio window into a 128x128 RGB array.

    Reproduces create_spectrogram_dataset.py exactly (mel -> power_to_db(ref=max)
    -> magma, axis off, tight crop), rendered to memory instead of disk.
    """
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
    # Model's first layer is Rescaling(1/255), so keep raw 0-255 values.
    return np.expand_dims(np.array(img).astype(np.float32), axis=0)


def main():
    parser = argparse.ArgumentParser(description="Real-time glass-break alarm.")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Probability at/above which to raise an alert (default 0.80).")
    parser.add_argument("--window", type=float, default=2.0,
                        help="Rolling window length in seconds (default 2.0).")
    parser.add_argument("--hop", type=float, default=1.0,
                        help="Seconds between analyses (default 1.0).")
    parser.add_argument("--silence", type=float, default=0.01,
                        help="RMS energy below which a window is treated as silence (default 0.01).")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the .keras model.")
    parser.add_argument("--device", type=int, default=None,
                        help="Input device index (see --list-devices).")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio input devices and exit.")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    window_len = int(args.window * FS)

    print("Loading model...")
    model = tf.keras.models.load_model(args.model)
    print(f"Model loaded: {args.model}")

    # Rolling ring buffer, filled by the audio callback.
    ring = np.zeros(window_len, dtype=np.float32)
    lock = threading.Lock()

    def callback(indata, frames, time_info, status):
        nonlocal ring
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono = indata[:, 0]
        n = len(mono)
        with lock:
            if n >= window_len:
                ring = mono[-window_len:].copy()
            else:
                ring = np.roll(ring, -n)
                ring[-n:] = mono

    print(
        f"\nListening... window={args.window}s hop={args.hop}s "
        f"threshold={args.threshold:.2f}\nPress Ctrl+C to stop.\n"
    )

    try:
        with sd.InputStream(samplerate=FS, channels=1, dtype="float32",
                            device=args.device, callback=callback):
            # Prime the buffer so the first analysis has a full window.
            time.sleep(args.window)
            while True:
                time.sleep(args.hop)
                with lock:
                    window = ring.copy()

                rms = float(np.sqrt(np.mean(window ** 2)))
                stamp = time.strftime("%H:%M:%S")

                if rms < args.silence:
                    print(f"[{stamp}] silence (rms={rms:.4f})")
                    continue

                arr = make_spectrogram_array(window, FS)
                # Classes were sorted alphabetically at training time
                # (glassbreak=0, other=1), so the sigmoid is P(other);
                # glass-break probability is its complement.
                prob = 1.0 - float(model(arr, training=False)[0][0])

                if prob >= args.threshold:
                    print(f"[{stamp}] 🚨 GLASS BREAK DETECTED!  prob={prob:.4f} rms={rms:.4f}")
                else:
                    print(f"[{stamp}] listening... prob={prob:.4f} rms={rms:.4f}")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

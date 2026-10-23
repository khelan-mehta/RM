import tensorflow as tf
import numpy as np
from PIL import Image
import librosa

# ===============================
# Settings
# ===============================

MODEL_PATH = "models/glass_detector_v2.keras"
IMAGE_PATH = "live_spectrogram.png"
IMG_SIZE = (128, 128)

# ===============================
# Check Silence
# ===============================

y, sr = librosa.load("live_audio.wav", sr=22050)

rms = np.sqrt(np.mean(y ** 2))

print(f"RMS Energy: {rms:.6f}")

if rms < 0.01:
    print("🔇 Silence detected.")
    exit()

print("Step 1 - RMS check passed")

# ===============================
# Load Model
# ===============================

print("Step 2 - Loading model")
model = tf.keras.models.load_model(MODEL_PATH)
print("Step 3 - Model loaded")

# ===============================
# Load Image
# ===============================

print("Step 4 - Loading image")

img = Image.open(IMAGE_PATH)
img = img.convert("RGB")
img = img.resize(IMG_SIZE)

img = np.array(img).astype(np.float32)
img = np.expand_dims(img, axis=0)

print("Step 5 - Image loaded")

# ===============================
# Prediction
# ===============================

print("Step 6 - Predicting")

# Classes were sorted alphabetically at training (glassbreak=0, other=1), so the
# sigmoid outputs P(other); glass-break probability is its complement.
prob = 1.0 - model.predict(img, verbose=0)[0][0]

print("Step 7 - Prediction complete")

print(f"\nGlass Probability: {prob:.4f}")

THRESHOLD = 0.80

if prob >= THRESHOLD:
    print("🚨 GLASS BREAK DETECTED!")
else:
    print("✅ No Glass Break")
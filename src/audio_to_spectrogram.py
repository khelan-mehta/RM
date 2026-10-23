import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

AUDIO_FILE = "live_audio.wav"
OUTPUT_IMAGE = "live_spectrogram.png"

# Load audio
y, sr = librosa.load(AUDIO_FILE, sr=22050)

# Create Mel Spectrogram
mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=128
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)

# Save spectrogram
plt.figure(figsize=(2.56, 2.56), dpi=50)

librosa.display.specshow(
    mel_db,
    sr=sr,
    cmap="magma"
)

plt.axis("off")

plt.savefig(
    OUTPUT_IMAGE,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print("Spectrogram saved as", OUTPUT_IMAGE)
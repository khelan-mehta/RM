import sounddevice as sd
import soundfile as sf
import numpy as np

FS = 22050
DURATION = 2

print("Recording starts in 3 seconds...")
sd.sleep(3000)

print("Recording...")

audio = sd.rec(
    int(FS * DURATION),
    samplerate=FS,
    channels=1,
    dtype=np.float32
)

sd.wait()

sf.write("live_audio.wav", audio, FS)

print("✅ Recording Complete!")
print("Saved as live_audio.wav")
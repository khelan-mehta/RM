from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

INPUT = Path("cnn_dataset/glassbreak")
OUTPUT = Path("augmented_glass")

OUTPUT.mkdir(exist_ok=True)

files = list(INPUT.glob("*.wav"))

print("Total Glass Files:", len(files))

count = 0

for file in files:

    audio, sr = librosa.load(file, sr=22050)

    # Original
    sf.write(OUTPUT / f"{count}_original.wav", audio, sr)

    # Time Shift
    shift_audio = np.roll(audio, int(0.2 * sr))
    sf.write(OUTPUT / f"{count}_shift.wav", shift_audio, sr)

    # Pitch Shift
    pitch_audio = librosa.effects.pitch_shift(
        audio,
        sr=sr,
        n_steps=2
    )
    sf.write(OUTPUT / f"{count}_pitch.wav", pitch_audio, sr)

    # Time Stretch
    stretch_audio = librosa.effects.time_stretch(
        audio,
        rate=1.1
    )
    sf.write(OUTPUT / f"{count}_stretch.wav", stretch_audio, sr)

    # Noise
    noise = np.random.normal(0, 0.005, len(audio))
    noise_audio = audio + noise
    sf.write(OUTPUT / f"{count}_noise.wav", noise_audio, sr)

    count += 1

    if count % 25 == 0:
        print(f"Processed {count}/{len(files)}")

print("\nDone!")
print(f"Original Glass Files : {len(files)}")
print(f"Total Augmented Files: {len(files) * 5}")
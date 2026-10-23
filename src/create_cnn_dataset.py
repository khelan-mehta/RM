import random
from pathlib import Path

import librosa
import soundfile as sf
import pandas as pd

# ==========================
# Load master dataset
# ==========================

df = pd.read_csv("master_dataset.csv")

OUTPUT = Path("cnn_dataset")

(OUTPUT / "glassbreak").mkdir(parents=True, exist_ok=True)
(OUTPUT / "other").mkdir(parents=True, exist_ok=True)

print("Total files:", len(df))

saved = 0

for index, row in df.iterrows():

    audio, sr = librosa.load(row["audio_path"], sr=22050)

    # ------------------------
    # Glass event
    # ------------------------

    if row["label"] == "glassbreak":

        start = int(row["start"] * sr)
        end = int(row["end"] * sr)

        clip = audio[start:end]

    # ------------------------
    # Other
    # ------------------------

    else:

        # DCASE background
        if pd.isna(row["start"]):

            length = 2 * sr

            if len(audio) > length:

                begin = random.randint(0, len(audio) - length)

                clip = audio[begin:begin + length]

            else:

                clip = audio

        # Baby cry / Gunshot
        else:

            start = int(row["start"] * sr)
            end = int(row["end"] * sr)

            clip = audio[start:end]

    # Save cropped clip

    out_file = OUTPUT / row["label"] / f"{saved}.wav"

    sf.write(out_file, clip, sr)

    saved += 1

    if saved % 250 == 0:

        print(saved, "files saved")

print("\nFinished!")

print("Saved clips:", saved)
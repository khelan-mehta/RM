from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ==========================
# Settings
# ==========================

CSV_FILES = [
    "train.csv",
    "validation.csv",
    "test.csv"
]

OUTPUT_FOLDERS = {
    "train.csv": "train_images",
    "validation.csv": "validation_images",
    "test.csv": "test_images"
}

IMG_SIZE = (128, 128)

# ==========================
# Process Each CSV
# ==========================

for csv_file in CSV_FILES:

    df = pd.read_csv(csv_file)

    output_root = Path(OUTPUT_FOLDERS[csv_file])

    (output_root / "glassbreak").mkdir(parents=True, exist_ok=True)
    (output_root / "other").mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {csv_file}")

    for i, row in df.iterrows():

        audio_path = row["path"]
        label = "glassbreak" if row["label"] == 1 else "other"

        try:

            y, sr = librosa.load(audio_path, sr=22050)

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=128
            )

            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )

            plt.figure(figsize=(2,2))

            librosa.display.specshow(
                mel_db,
                sr=sr,
                cmap="magma"
            )

            plt.axis("off")

            save_path = output_root / label / f"{i}.png"

            plt.savefig(
                save_path,
                bbox_inches="tight",
                pad_inches=0
            )

            plt.close()

        except Exception as e:
            print("Error:", audio_path)
            print(e)

        if (i + 1) % 250 == 0:
            print(f"{i+1}/{len(df)}")

print("\nDone!")
from pathlib import Path
import pandas as pd

# ===========================
# Paths
# ===========================

DCASE_ROOT = Path(
    r"M:\RM project\dataset\DCASE2017\TUT-rare-sound-events-2017-development\data\mixture_data\devtrain"
)

ESC_ROOT = Path(
    r"M:\RM project\dataset\ESC-50-master"
)

dataset = []

# ===========================
# DCASE
# ===========================

meta_files = list(DCASE_ROOT.rglob("event_list_*.csv"))

print(f"Found {len(meta_files)} DCASE metadata files")

for meta in meta_files:

    df = pd.read_csv(
        meta,
        sep="\t",
        header=None,
        names=[
            "filename",
            "start",
            "end",
            "event"
        ]
    )

    audio_folder = meta.parent.parent / "audio"

    for _, row in df.iterrows():

        audio_path = audio_folder / row["filename"]

        if pd.isna(row["event"]):

            label = "other"

            start = None
            end = None

        elif row["event"] == "glassbreak":

            label = "glassbreak"

            start = row["start"]
            end = row["end"]

        else:

            label = "other"

            start = row["start"]
            end = row["end"]

        dataset.append({

            "audio_path": str(audio_path),

            "label": label,

            "start": start,

            "end": end,

            "source": "DCASE"

        })

# ===========================
# ESC50
# ===========================

esc = pd.read_csv(
    ESC_ROOT / "meta" / "esc50.csv"
)

for _, row in esc.iterrows():

    audio_path = ESC_ROOT / "audio" / row["filename"]

    if row["category"] == "glass_breaking":
        label = "glassbreak"
    else:
        label = "other"

    dataset.append({

        "audio_path": str(audio_path),

        "label": label,

        "start": 0,

        "end": 5,

        "source": "ESC50"

    })

dataset = pd.DataFrame(dataset)

print("\nDataset Summary\n")

print(dataset["label"].value_counts())

print("\nSources\n")

print(dataset["source"].value_counts())

print("\nColumns\n")

print(dataset.columns)

dataset.to_csv(
    "master_dataset.csv",
    index=False
)

print("\nSaved as master_dataset.csv")

print("Total Samples:", len(dataset))
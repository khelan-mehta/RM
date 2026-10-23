from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------
# Paths
# -------------------------

GLASS = Path("augmented_glass")
OTHER = Path("cnn_dataset/other")

# -------------------------
# Read Files
# -------------------------

glass_files = list(GLASS.glob("*.wav"))
other_files = list(OTHER.glob("*.wav"))

print("Glass :", len(glass_files))
print("Other :", len(other_files))

# -------------------------
# Create DataFrame
# -------------------------

data = []

for file in glass_files:
    data.append({
        "path": str(file),
        "label": 1
    })

for file in other_files:
    data.append({
        "path": str(file),
        "label": 0
    })

df = pd.DataFrame(data)

print("\nTotal Samples:", len(df))

print("\nClass Distribution")

print(df["label"].value_counts())

# -------------------------
# Train + Temp
# -------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# -------------------------
# Validation + Test
# -------------------------

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

print("\nTrain :", len(train_df))
print("Validation :", len(val_df))
print("Test :", len(test_df))

print("\nTrain Distribution")
print(train_df["label"].value_counts())

print("\nValidation Distribution")
print(val_df["label"].value_counts())

print("\nTest Distribution")
print(test_df["label"].value_counts())

# -------------------------
# Save CSV
# -------------------------

train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\nCSV files saved.")
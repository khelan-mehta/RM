import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

IMG_SIZE = (128, 128)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "test_images",
    image_size=IMG_SIZE,
    batch_size=32,
    shuffle=False
)

model = tf.keras.models.load_model(
    "models/glass_detector_v2.keras"
)

# Predict
y_prob = model.predict(test_ds)
y_pred = (y_prob > 0.5).astype(int).flatten()

# True labels
y_true = np.concatenate(
    [y.numpy() for _, y in test_ds],
    axis=0
)

print("\nClassification Report\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["glassbreak", "other"]
    )
)

print("\nConfusion Matrix\n")
print(confusion_matrix(y_true, y_pred))
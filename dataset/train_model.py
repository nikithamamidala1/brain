import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pickle

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = r"C:\Users\Rana\Desktop\brain tumor\dataset"  # Make sure path exists
MODEL_PATH = "brain_tumor_model.h5"

# ----------------------------
# DATA AUGMENTATION
# ----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# ----------------------------
# LOAD DATASET
# ----------------------------
train_path = os.path.join(DATA_DIR, "train")
val_path = os.path.join(DATA_DIR, "validation")
test_path = os.path.join(DATA_DIR, "test")

# Check paths exist
for p in [train_path, val_path, test_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"❌ Folder not found: {p}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ----------------------------
# NORMALIZATION + AUGMENTATION
# ----------------------------
normalization = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization(x)), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))
test_ds = test_ds.map(lambda x, y: (normalization(x), y))

# Prefetch for speed
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ---

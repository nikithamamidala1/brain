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
DATA_DIR = r"C:\Users\Rana\Desktop\brain tumor\dataset"  # Update path if needed
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
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
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

# ----------------------------
# CALCULATE CLASS WEIGHTS
# ----------------------------
# Extract labels from train dataset
train_labels = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
class_weights_values = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = {i: w for i, w in enumerate(class_weights_values)}
print("Class weights:", class_weights)

# ----------------------------
# BUILD MODEL (MobileNetV2)
# ----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

# Fine-tuning: unfreeze last 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")  # binary classification
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # small learning rate for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# TRAIN MODEL
# ----------------------------
print("🚀 Training started...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights
)

# ----------------------------
# EVALUATE MODEL
# ----------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save(MODEL_PATH)
print(f"💾 Model saved as {MODEL_PATH}")

# ----------------------------
# SAVE TRAINING HISTORY
# ----------------------------
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("📈 Training history saved as history.pkl")

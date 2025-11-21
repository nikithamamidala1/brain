import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

model = load_model()

# Define class names (must match your dataset folders)
class_names = ["No Tumor", "Tumor"]

# Streamlit UI
st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image, and the model will predict if a tumor is present.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]  # Example: [0.10, 0.90]

    prob_no_tumor = float(prediction[0])
    prob_tumor = float(prediction[1])

    # Final result
    result = "🧠 Tumor Detected" if prob_tumor > prob_no_tumor else "✅ No Tumor"

    # Show result + confidence
    st.subheader("Prediction Result:")
    st.success(f"{result} (Confidence: {max(prob_tumor, prob_no_tumor)*100:.2f}%)")

    # Bar chart visualization
    st.subheader("Prediction Probabilities")
    fig, ax = plt.subplots()
    bars = ax.bar(class_names, [prob_no_tumor, prob_tumor])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])

    # Add % labels
    for bar, prob in zip(bars, [prob_no_tumor, prob_tumor]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f"{prob*100:.2f}%", ha="center", fontsize=10)

    st.pyplot(fig)

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from keras.models import load_model
from waste_type_mapping import waste_type_mapping

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Waste Classification System",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ AI-Based Waste Classification System")
st.write(
    "Upload an image of waste and the system will predict "
    "its **class** and **waste type**."
)

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_trained_model():
    return load_model("waste_classification_model.h5")

model = load_trained_model()

# --------------------------------------------------
# Load class names
# --------------------------------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# --------------------------------------------------
# File uploader
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
if uploaded_file is not None:
    # Display image
    img = tf.keras.preprocessing.image.load_img(
        uploaded_file, target_size=(224, 224)
    )
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    predicted_class = class_names[class_index]
    waste_type = waste_type_mapping.get(predicted_class, "Unknown")

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------
    st.success(f"🧠 Predicted Class: **{predicted_class}**")
    st.info(f"🗑️ Waste Type: **{waste_type}**")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")

    # Color-coded output
    if waste_type == "Recyclable":
        st.markdown("### ♻️ Please place this in the **Recyclable Bin**")
    elif waste_type == "Food":
        st.markdown("### 🍃 Please place this in the **Organic / Compost Bin**")
    elif waste_type == "Hazardous":
        st.markdown("### ☠️ Handle with care – **Hazardous Waste**")
    else:
        st.markdown("### 🗑️ Please place this in the **Residual Waste Bin**")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "SDG-12: Responsible Consumption & Production | "
    "AI for Sustainable Waste Management"
)

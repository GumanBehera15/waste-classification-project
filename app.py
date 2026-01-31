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
st.caption(
    "Upload a waste image to predict its **class**, **waste type**, and **confidence**."
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
# Layout: Two columns
# --------------------------------------------------
left_col, right_col = st.columns([1, 1])

# --------------------------------------------------
# File uploader (LEFT)
# --------------------------------------------------
with left_col:
    uploaded_file = st.file_uploader(
        "📤 Upload waste image",
        type=["jpg", "jpeg", "png"]
    )

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
if uploaded_file is not None:
    # Load & display image
    img = tf.keras.preprocessing.image.load_img(
        uploaded_file, target_size=(224, 224)
    )

    with left_col:
        st.image(img, caption="Uploaded Image", width=260)

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
    # Display results (RIGHT)
    # --------------------------------------------------
    with right_col:
        st.subheader("🔍 Prediction Results")

        st.metric(
            label="Predicted Class",
            value=predicted_class
        )

        st.metric(
            label="Waste Type",
            value=waste_type
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        # Color-coded guidance
        if waste_type == "Recyclable":
            st.success("♻️ Place this in the **Recyclable Bin**")
        elif waste_type == "Food":
            st.success("🍃 Place this in the **Organic / Compost Bin**")
        elif waste_type == "Hazardous":
            st.error("☠️ **Hazardous Waste** – Handle with care")
        else:
            st.info("🗑️ Place this in the **Residual Waste Bin**")

# --------------------------------------------------
# Footer (compact)
# --------------------------------------------------
st.markdown("---")
st.caption("SDG-12: Responsible Consumption & Production | AI for Sustainability")

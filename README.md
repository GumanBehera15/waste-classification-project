# ♻️ Waste Classification System

An AI-powered waste classification system that automatically identifies the type of waste from an image using deep learning.  
The project aims to improve waste segregation efficiency and support sustainable waste management practices.

---

## 📌 Project Overview

Improper waste segregation leads to recyclable materials ending up in landfills, increasing environmental pollution and recycling costs. Manual sorting is slow, inconsistent, and resource-intensive.

This project addresses the problem by using **computer vision and deep learning** to classify waste images into predefined categories, enabling faster and more reliable waste segregation.

---

## ⚙️ How the System Works

1. **Image Input**  
   The user provides an image of a waste item.

2. **Image Preprocessing**  
   The image is resized and normalized to match the model’s input requirements.

3. **Model Inference**  
   A trained Convolutional Neural Network (CNN) processes the image and predicts the waste category.

4. **Result Mapping**  
   The predicted class index is mapped to a human-readable waste category using predefined class mappings.

5. **Output Display**  
   The predicted waste type is displayed to the user through the application interface.

---

## 🚀 Features
- 📷 Image-based waste classification  
- 🧠 Deep learning model trained on waste images  
- ⚡ Fast and automated prediction  
- 🧩 Easy-to-use application interface  
- ♻️ Supports sustainable waste management  

---

## 🧠 Model Details
- Model Type: **Convolutional Neural Network (CNN)**
- Framework: **TensorFlow / Keras**
- Input: Waste image
- Output: Waste category label
- Model Format: `.h5`

---

## 🧰 Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV / PIL  
- Streamlit (for UI)  
- Jupyter Notebook  

---

## 📂 Project Structure
```text
waste-classification-project/
│
├── app.py                         # Application entry point
├── Waste Classification.ipynb     # Model training notebook
├── waste_classification_model.h5  # Trained model
├── class_names.json               # Class label definitions
├── waste_type_mapping.py          # Mapping logic for predictions
├── requirement.txt                # Project dependencies
└── README.md

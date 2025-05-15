import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import uuid
import os

# Setup folder upload
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model('model_klasifikasi_penyakit_jagung_resnet50v2tunning.h5')
IMG_SIZE = (256, 256)
CLASSES = ['Unknown', 'Bercak Daun', 'Hawar Daun', 'Karat Daun', 'Sehat']

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #c2f0f7, #fcdff1);
    }
    .main {
        background: transparent;
    }
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
    }
    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
    .stButton > button {
        font-size: 16px;
        padding: 0.6em 1.5em;
        border-radius: 12px;
        border: 2px solid #444;
        background-color: white;
        font-weight: bold;
    }
    .image-preview {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- UI Header ---
st.markdown("<h1>Corn Leaf Classification</h1>", unsafe_allow_html=True)
st.markdown('<div class="button-container">', unsafe_allow_html=True)

# Upload & Process buttons
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
process_button = st.button("Proses")

st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Logic ---
if uploaded_file is not None and process_button:
    filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preview image
    image = Image.open(filepath)
    st.markdown('<div class="image-preview">', unsafe_allow_html=True)
    st.image(image, caption="Gambar yang Diupload", use_column_width=False, width=300)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    label = CLASSES[predicted_index]
    confidence = round(prediction[0][predicted_index] * 100, 2)

    # Output
    st.markdown(f"<h3>Hasil Prediksi: <span style='color:#008000'>{label}</span></h3>", unsafe_allow_html=True)
    st.markdown(f"<h4>Tingkat Keyakinan: {confidence}%</h4>", unsafe_allow_html=True)

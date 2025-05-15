import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import uuid
import os

# Setup folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model('model_klasifikasi_penyakit_jagung_resnet50v2tunning.h5')
IMG_SIZE = (256, 256)
CLASSES = ['Unknown', 'Bercak Daun', 'Hawar Daun', 'Karat Daun', 'Sehat']

# Custom CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        color: black;
    }
    .result {
        background-color: #fff8dc;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 18px;
        color: black;
        text-align: center;
    }
    .upload-button {
        background-color: transparent;
        border: 2px solid black;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 500;
        color: black;
        cursor: pointer;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .upload-button:hover {
        background-color: goldenrod;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>Corn Leaf Classification</h1>", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("Upload gambar daun jagung", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tampilkan gambar
    image = Image.open(filepath)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    label = CLASSES[predicted_index]
    confidence = round(prediction[0][predicted_index] * 100, 2)

    # Hasil
    st.markdown(f"""
    <div class='result'>
        <p><strong>Hasil Prediksi:</strong> {label}</p>
        <p><strong>Tingkat Keyakinan:</strong> {confidence}%</p>
    </div>
    """, unsafe_allow_html=True)

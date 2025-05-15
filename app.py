import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import uuid
import os

# Setup upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model('model_klasifikasi_penyakit_jagung_resnet50v2tunning.h5')
IMG_SIZE = (256, 256)
CLASSES = ['Unknown', 'Bercak Daun', 'Hawar Daun', 'Karat Daun', 'Sehat']

# Streamlit config
st.set_page_config(page_title="CornAI - Klasifikasi Daun Jagung", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #c2f0f7, #fcdff1);
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        padding-top: 50px;
    }
    .title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        color: #222;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        margin-top: -20px;
        color: #555;
    }
    .btn-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 30px;
        margin-bottom: 50px;
    }
    .custom-button {
        background-color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.3s ease-in-out;
    }
    .custom-button:hover {
        background-color: #f0f0f0;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Corn Leaf Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Klasifikasi Penyakit Daun Jagung Secara Otomatis</div>', unsafe_allow_html=True)

# Tombol Coba Sekarang & Informasi
col1, col2 = st.columns(2)
with col1:
    coba = st.button("🌽 Coba Sekarang", use_container_width=True)
with col2:
    if st.button("ℹ️ Informasi", use_container_width=True):
        st.info("Aplikasi ini menggunakan model deep learning (ResNet50V2) untuk mengklasifikasikan penyakit daun jagung. Upload gambar daun dan dapatkan hasilnya.")

# Form Upload dan Prediksi
if coba:
    uploaded_file = st.file_uploader("Upload gambar daun jagung (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Simpan file
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
        st.success(f"Hasil Prediksi: **{label}**")
        st.info(f"Tingkat Keyakinan: **{confidence}%**")

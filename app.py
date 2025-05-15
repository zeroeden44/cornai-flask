import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import uuid
import os

# Setup folder untuk simpan gambar sementara
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model('model_klasifikasi_penyakit_jagung_resnet50v2tunning.h5')
IMG_SIZE = (256, 256)
CLASSES = ['Unknown', 'Bercak Daun', 'Hawar Daun', 'Karat Daun', 'Sehat']

# -------------------- CSS Styling --------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #a8edea, #fed6e3) !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: transparent;
    }
    h1 {
        text-align: center;
        color: #333;
    }
    .uploaded-img {
        text-align: center;
        margin-top: 20px;
    }
    .result-box {
        background-color: #fff8dc;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .result-box h3 {
        color: #333;
    }
    .stButton>button {
        background-color: goldenrod;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Judul --------------------
st.markdown("<h1>Klasifikasi Penyakit Daun Jagung 🌽</h1>", unsafe_allow_html=True)

# -------------------- Upload --------------------
uploaded_file = st.file_uploader("Upload gambar daun jagung", type=["jpg", "jpeg", "png"])

# -------------------- Proses Prediksi --------------------
if uploaded_file is not None:
    filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(filepath)

    # Tampilkan gambar yang diunggah
    st.markdown('<div class="uploaded-img">', unsafe_allow_html=True)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocessing
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    label = CLASSES[predicted_index]
    confidence = round(prediction[0][predicted_index] * 100, 2)

    # Hasil prediksi
    st.markdown(f"""
        <div class="result-box">
            <h3>🧪 Hasil Prediksi: <span style="color:darkgreen;">{label}</span></h3>
            <h3>📊 Tingkat Keyakinan: <span style="color:darkblue;">{confidence}%</span></h3>
        </div>
    """, unsafe_allow_html=True)

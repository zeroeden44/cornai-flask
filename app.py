import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

st.set_page_config(page_title="Corn Leaf Classification", layout="centered")

# Cache model agar cuma load sekali
@st.cache_resource
def load_my_model():
    model = load_model(r"D:\sekripsi\cornai-flask\model_klasifikasi_penyakit_jagung_resnet50v2tunning.h5")
    return model

model = load_my_model()

class_names = ['Unknown', 'Bercak Daun', 'Hawar Daun', 'Karat Daun', 'Sehat']

def predict_image(image_pil):
    img = image_pil.resize((256, 256))
    img_array = np.array(img) / 255.0

    # Jika gambar PNG ada alpha channel, hapus
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)  # batch dim

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx] * 100
    predicted_label = class_names[class_idx]

    return predicted_label, confidence

# Styling sederhana
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #a8edea, #fed6e3);
    font-family: 'Segoe UI', sans-serif;
}
.result-box {
    background-color: #fff8dc;
    color: black;
    padding: 10px 20px;
    border-radius: 10px;
    margin-top: 20px;
    font-size: 18px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🌽 Corn Leaf Classification")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Preview Gambar", use_column_width=True)

    if st.button("Proses"):
        label, conf = predict_image(image)
        st.markdown(f"""
        <div class="result-box">
            <p><strong>Hasil Prediksi:</strong> {label}</p>
            <p><strong>Confidence:</strong> {conf:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

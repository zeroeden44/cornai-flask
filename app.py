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

st.title("Klasifikasi Penyakit Daun Jagung")

uploaded_file = st.file_uploader("Upload gambar daun jagung", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Simpan file sementara
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

    # Hasil prediksi
    st.markdown(f"### Hasil Prediksi: `{label}`")
    st.markdown(f"### Tingkat Keyakinan: `{confidence}%`")

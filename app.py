from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = load_model('model_klasifikasi_penyakit_jagung_resnet50v2tunning.h5')
IMG_SIZE = (256, 256)
CLASSES = ['Unknown', 'Bercak Daun', 'Hawar Daun','Karat Daun','Sehat']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('klasifikasi.html', error="Tidak ada file yang diunggah.")

        file = request.files['image']
        if file.filename == '':
            return render_template('klasifikasi.html', error="Tidak ada file yang dipilih.")

        # Simpan file
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocessing gambar
        img = load_img(filepath, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalisasi sesuai model training

        # Prediksi
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        label = CLASSES[predicted_index]
        confidence = round(prediction[0][predicted_index] * 100, 2)

        return render_template('klasifikasi.html',
                               label=label,
                               confidence=confidence,
                               image_url=filepath)

    return render_template('klasifikasi.html')

if __name__ == '__main__':
    app.run(debug=True)

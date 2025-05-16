import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image, ImageOps

# === CONFIGURATION ===
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/best_model.tflite'
TARGET_SIZE = (128, 128)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TFLite model & allocate tensors :contentReference[oaicite:0]{index=0}
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path):
    # Load, convert to grayscale, crop, resize, normalize
    img = Image.open(path).convert('L')
    bw  = img.point(lambda x: 0 if x<200 else 255, '1')
    if (bbox := bw.getbbox()):
        img = img.crop(bbox)
    img = img.resize(TARGET_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Match interpreter’s expected shape
    return np.expand_dims(arr, axis=(0, -1))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)  # handle uploads in Flask :contentReference[oaicite:1]{index=1}
            
            # Prepare input & run inference
            input_data = preprocess_image(path)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])
            
            # Extract top prediction
            idx   = np.argmax(preds[0])
            conf  = preds[0][idx]
            label = str(idx)  # replace with your class mapping if available
            
            return render_template('index.html',
                                   filename=filename,
                                   prediction=label,
                                   confidence=f"{conf:.2%}")
        return redirect(request.url)
    return render_template('index.html')

# To serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    app.run(debug=True)

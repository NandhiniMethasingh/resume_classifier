from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import PyPDF2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load LSTM model and preprocessing tools
model = tf.keras.models.load_model('model/lstm_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return "No file uploaded", 400

    file = request.files['resume']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Extract text from PDF
    resume_text = ''
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            resume_text += page.extract_text()

    # Preprocess text
    sequence = tokenizer.texts_to_sequences([resume_text])
    padded = pad_sequences(sequence, maxlen=200)

    # Make prediction
    prediction = model.predict(padded)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    return render_template('result.html', prediction=predicted_label[0], filename=filename)

#if __name__ == '__main__':
 #   app.run(debug=True)
# Main entry point
if __name__ == '__main__':
    # This is used only when running locally, Render uses gunicorn to serve the app
    app.run(host='0.0.0.0', port=10000, debug=True)

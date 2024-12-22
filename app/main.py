from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join("saved_models", "tumor_model.h5")
model = load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Preprocess the image
    img = preprocess_image(file)
    prediction = model.predict(img)
    result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

    return jsonify({'result': result, 'confidence': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)

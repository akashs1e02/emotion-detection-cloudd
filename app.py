from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
from keras.models import load_model

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "emotion_model.h5")
model = load_model(model_path, compile=False)

emotion_labels = ["Angry", "Happy", "Sad"]

@app.route("/")
def home():
    return "Emotion Detection Running on Cloud!"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    face = cv2.resize(img, (64, 64))
    face = face / 255.0
    face = np.reshape(face, (1, 64, 64, 1))

    prediction = model.predict(face)
    emotion = emotion_labels[np.argmax(prediction)]

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load model và nhãn
model = tf.keras.models.load_model('keras_model.h5')
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.asarray(img) / 255.0
    prediction = model.predict(np.expand_dims(img_array, axis=0))[0]
    index = np.argmax(prediction)
    return labels[index], float(prediction[index])

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        image = request.files["image"]
        path = os.path.join("static", "uploaded_image.jpg")
        image.save(path)
        label, confidence = predict_image(path)
        result = f"{label} (độ tin cậy: {confidence:.2%})"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

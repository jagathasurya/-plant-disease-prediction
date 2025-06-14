
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('disease_prediction_model.h5')
class_names = sorted(os.listdir('PlantVillage'))

@app.route('/')
def home():
    return render_template('index.html')

from io import BytesIO

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img = image.load_img(BytesIO(img_file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({'prediction': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)

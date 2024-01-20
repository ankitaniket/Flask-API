import os
import glob
from PIL import Image
import tensorflow as tf
from flask import Flask, request,jsonify
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("best_model.h5")

# platform Darwin-22.4.0-x86_64-i386-64bit

# python version 3.9.16
# environment ==>> conda ==> tf


@app.route('/', methods=['POST'])
def predict():
    # Get the uploaded image file
    image_file = request.files['image']

    # Load the model

    # Predict & classify image
    img = Image.open(image_file).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    input_arr = np.array([img_array])
    pred = np.argmax(model.predict(input_arr))

    if pred == 0:
        result = "This image does not contain a pothole."
    else:
        result = "This image contains a pothole."

    # Add the Access-Control-Allow-Origin header to the response
    response = jsonify({'result': result})
    response.headers.add('Access-Control-Allow-Origin', '*')

    # Return the prediction result as a JSON object
    return response


if __name__ == '__main__':
    app.run(debug=True,port=7000)
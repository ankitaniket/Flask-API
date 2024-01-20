import tensorflow as tf
from flask import Flask, request,jsonify
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from io import BytesIO
import requests
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("Final best_model-2.h5")

@app.errorhandler(Exception)
def handle_error(e):
    app.logger.error(str(e))
    response = jsonify({'error': str(e)})
    response.status_code = 500
    return response

@app.route('/', methods=['POST'])
def predict():
    if request.headers['Content-Type'] == 'application/json':
        # If the request is in JSON format
        data = request.get_json()
        image_url = data.get('image_url', '')
    else:

        # If the request is in form data format

        image_file = request.files.get('image', None)
        if image_file:
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

            response = make_response(jsonify({'result': result}), 200)
            print("This is result - ",result)
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
        else:
            return jsonify({'error': 'Image file not provided'}), 400

    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    input_arr = np.array([img_array])
    pred = np.argmax(model.predict(input_arr))

    if pred == 0:
        result = "This image does not contain a pothole."
    else:
        result = "This image contains a pothole."

    response = make_response(jsonify({'result': result}), 200)
    return response

if __name__ == '__main__':
    app.run(debug=True, port=8001)
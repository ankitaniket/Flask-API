import tensorflow as tf
import os
from flask import Flask, request,jsonify,render_template,make_response,render_template_string
from PIL import Image
import numpy as np
from io import BytesIO
import requests
from keras.applications.inception_v3 import preprocess_input
from flask_cors import CORS
import uuid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


app = Flask(__name__)
CORS(app)

# Python 3.9.18
# env conda
model = tf.keras.models.load_model("Final best_model-2.h5")

# Your Firebase configuration
@app.errorhandler(Exception)
def handle_error(e):
    app.logger.error(str(e))
    response = jsonify({'error': str(e)})
    response.status_code = 500
    return response


def generate_unique_filename(result):
    prefix = 'Y' if result == "This image contains a pothole." else ''
    return f"{prefix}{uuid.uuid4()}"  

@app.route('/pred', methods=['POST'])
def predict():
    if request.headers['Content-Type'] == 'application/json':

        data = request.get_json()
        image_url = data.get('image_url', '')
    else:
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
            print("This is result - ", result)


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

@app.route('/whoami')
def whoami():
    # Fetch the IP address using ipify.org
    ipify_response = requests.get("https://api.ipify.org/?format=json")
    ipify_data = ipify_response.json()
    
    geo_response = requests.get(f"http://www.geoplugin.net/json.gp?ip={ipify_data['ip']}")
    geo_data = geo_response.json()
    
    # Format the output with styling for the terminal
    output = """
    \033[1mIP Address:\033[0m {ip}
    \033[1mCity:\033[0m {city}
    \033[1mRegion:\033[0m {region} (\033[36m{region_code}\033[0m)
    \033[1mCountry:\033[0m {country} (\033[36m{country_code}\033[0m)
    \033[1mContinent:\033[0m {continent} (\033[36m{continent_code}\033[0m)
    \033[1mLatitude:\033[0m {latitude}
    \033[1mLongitude:\033[0m {longitude}
    \033[1mTimezone:\033[0m {timezone}
    \033[1mCurrency:\033[0m {currency} (\033[36m{currency_symbol}\033[0m)
    """.format(
        ip=geo_data['geoplugin_request'],
        city=geo_data['geoplugin_city'],
        region=geo_data['geoplugin_region'],
        region_code=geo_data['geoplugin_regionCode'],
        country=geo_data['geoplugin_countryName'],
        country_code=geo_data['geoplugin_countryCode'],
        continent=geo_data['geoplugin_continentName'],
        continent_code=geo_data['geoplugin_continentCode'],
        latitude=geo_data['geoplugin_latitude'],
        longitude=geo_data['geoplugin_longitude'],
        timezone=geo_data['geoplugin_timezone'],
        currency=geo_data['geoplugin_currencyCode'],
        currency_symbol=geo_data['geoplugin_currencySymbol']
    )
    
    return output


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    return render_template_string('Yepp, your curl command is running successfully for the test route. 😄')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
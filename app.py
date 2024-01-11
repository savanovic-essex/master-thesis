# Import necessary libraries
from flask import Flask, request, jsonify  # Flask is a micro web framework for Python.
import pickle  # Used for loading the serialized Python object (model).
import numpy as np  # NumPy is a library for array operations.
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables.
from flask_cors import CORS  # This is for handling Cross-Origin Resource Sharing (CORS).

app = Flask(__name__)  # Initialize a Flask application.
CORS(app)  # Enable CORS for the Flask app. This makes the API accessible to web applications.

# Initialize and load the LabelEncoder which is used to convert categorical features to numeric.
# This is necessary because machine learning models require numerical input.
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy', allow_pickle=True)  # Load the classes used by the LabelEncoder.

def encode_features(input_data, encoder):
    categorical_features = [
        'IM_INCIDENT_KEY', 'FIRE_BOX', 'STREET_HIGHWAY', 'ZIP_CODE',
        'INCIDENT_TYPE_DESC', 'PROPERTY_USE_DESC', 'BOROUGH_DESC', 'FLOOR'
    ]
    encoded_data = input_data.copy()  # Make a copy of the input data.
    # Iterate over the categorical features and encode them.
    for feature in categorical_features:
        if encoded_data[feature] != "":
            if encoded_data[feature] in encoder.classes_:
                encoded_value = encoder.transform([encoded_data[feature]])
                encoded_data[feature] = encoded_value[0]
            else:
                encoded_data[feature] = -1
        else:
            encoded_data[feature] = -1
    return encoded_data  # Return the encoded data.


# Load the LightGBM model from the serialized file.
model_file_path = 'lgbm_model.pkl'
with open(model_file_path, 'rb') as file:  # Open the file in read-binary mode.
    lgbm_model = pickle.load(file)  # Load the trained model.

@app.route('/')
def home():
    return "LightGBM Model API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  # Get JSON data from the POST request.

    try:
        # Encode features and make prediction
        encoded_inputs = encode_features(data, encoder)
        features = np.array([list(encoded_inputs.values())])
        prediction = lgbm_model.predict(features)
        return jsonify(prediction.tolist())

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

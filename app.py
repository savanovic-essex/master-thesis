# app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Initialize and load the LabelEncoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy', allow_pickle=True)

def encode_features(input_data, encoder):
    categorical_features = ['IM_INCIDENT_KEY', 'FIRE_BOX', 'STREET_HIGHWAY', 'ZIP_CODE',
                            'INCIDENT_TYPE_DESC', 'PROPERTY_USE_DESC', 'BOROUGH_DESC', 'FLOOR']
    encoded_data = input_data.copy()
    for feature in categorical_features:
        if encoded_data[feature] != "":
            if encoded_data[feature] in encoder.classes_:
                encoded_value = encoder.transform([encoded_data[feature]])
                encoded_data[feature] = encoded_value[0]
            else:
                encoded_data[feature] = -1
        else:
            encoded_data[feature] = -1
    return encoded_data


# Load the LightGBM model
model_file_path = 'best_lgbm_model_no_outliers.pkl'
with open(model_file_path, 'rb') as file:
    lgbm_model = pickle.load(file)


@app.route('/')
def home():
    return "LightGBM Model API"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        encoded_inputs = encode_features(data, encoder)
        features = np.array([list(encoded_inputs.values())])
        prediction = lgbm_model.predict(features)
        return jsonify(prediction.tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)

import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Header for your app
st.header("Incident Unit Prediction App")
st.text_input("Enter your Name: ", key="name")

# Load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy', allow_pickle=True)

# Load model
best_lgbm_model = joblib.load('best_lgbm_model.pkl')

st.subheader("Please enter the incident details for prediction")

# Create user input fields
input_IM_INCIDENT_KEY = st.text_input("IM Incident Key")
input_FIRE_BOX = st.text_input("Fire Box")
input_INCIDENT_TYPE_DESC = st.text_input("Incident Type Description")
input_PROPERTY_USE_DESC = st.text_input("Property Use Description")
input_STREET_HIGHWAY = st.text_input("Street/Highway")
input_ZIP_CODE = st.text_input("Zip Code")
input_BOROUGH_DESC = st.text_input("Borough Description")
input_FLOOR = st.text_input("Floor")
input_INCIDENT_MONTH = st.number_input("Incident Month", min_value=1, max_value=12, step=1)
input_INCIDENT_DAY_OF_WEEK = st.number_input("Incident Day of the Week", min_value=1, max_value=7, step=1)
input_INCIDENT_HOUR = st.number_input("Incident Hour", min_value=0, max_value=23, step=1)
input_RESPONSE_TIME = st.number_input("Response Time", min_value=0)
input_RESPONSE_TIME_MINUTES = st.number_input("Response Time (Minutes)", min_value=0.0, step=0.1)
input_YEAR = st.number_input("Year", min_value=2000, max_value=2023, step=1)
input_LATITUDE = st.number_input("Latitude", format="%.8f")
input_LONGITUDE = st.number_input("Longitude", format="%.8f")

# Assume you have a function to encode categorical features
def encode_features(input_data, encoder):
    # Define the categorical features that need encoding
    categorical_features = ['IM_INCIDENT_KEY', 'FIRE_BOX', 'STREET_HIGHWAY', 'ZIP_CODE',
                            'INCIDENT_TYPE_DESC', 'PROPERTY_USE_DESC', 'BOROUGH_DESC', 'FLOOR']

    # Copy input data to avoid modifying the original
    encoded_data = input_data.copy()

    # Apply label encoding to each categorical feature
    for feature in categorical_features:
        if encoded_data[feature] != "":
            encoded_value = encoder.transform([encoded_data[feature]])
            encoded_data[feature] = encoded_value[0]
        else:
            # Handle missing or empty values as needed
            encoded_data[feature] = -1  # Example: using -1 for missing values

    return encoded_data


if st.button('Make Prediction'):
    # Collect user inputs
    user_input = {
        'IM_INCIDENT_KEY': input_IM_INCIDENT_KEY,
        'FIRE_BOX': input_FIRE_BOX,
        'STREET_HIGHWAY': input_STREET_HIGHWAY,
        'ZIP_CODE': input_ZIP_CODE,
        'INCIDENT_TYPE_DESC': input_INCIDENT_TYPE_DESC,
        'PROPERTY_USE_DESC': input_PROPERTY_USE_DESC,
        'BOROUGH_DESC': input_BOROUGH_DESC,
        'FLOOR': input_FLOOR,
        'INCIDENT_MONTH': input_INCIDENT_MONTH,
        'INCIDENT_DAY_OF_WEEK': input_INCIDENT_DAY_OF_WEEK,
        'INCIDENT_HOUR': input_INCIDENT_HOUR,
        'RESPONSE_TIME': input_RESPONSE_TIME,
        'RESPONSE_TIME_MINUTES': input_RESPONSE_TIME_MINUTES,
        'YEAR': input_YEAR,
        'LATITUDE': input_LATITUDE,
        'LONGITUDE': input_LONGITUDE
    }

    # Encode categorical features
    encoded_inputs = encode_features(user_input, encoder)

    # Format inputs for prediction (ensure correct order and data types)
    # Example: inputs = np.array([list(encoded_inputs.values())])
    inputs = np.array([list(encoded_inputs.values())])

    # Make prediction
    prediction = best_lgbm_model.predict(inputs)
    st.write(f"Predicted number of units needed: {np.squeeze(prediction, -1)}")

    st.write(f"Thank you {st.session_state.name}! I hope you found this useful.")
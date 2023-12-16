import pandas as pd
import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("incidents.csv")

# String data types
# These columns are identifiers or categorical features with alphanumeric entries,
# hence they are converted to string (object) data type for accurate representation and manipulation.
df['IM_INCIDENT_KEY'] = df['IM_INCIDENT_KEY'].astype('str')  # Incident key as a string
df['FIRE_BOX'] = df['FIRE_BOX'].astype('str')  # Fire box number as a string
df['STREET_HIGHWAY'] = df['STREET_HIGHWAY'].astype('str')  # Street or highway name as a string
df['ZIP_CODE'] = df['ZIP_CODE'].astype('str')  # Zip code as a string

# Categorical data types
# These columns represent categories and are converted to the 'category' data type,
# which is memory efficient and useful for analysis that involves categorical data.
df['INCIDENT_TYPE_DESC'] = df['INCIDENT_TYPE_DESC'].astype('category')  # Description of incident type
df['PROPERTY_USE_DESC'] = df['PROPERTY_USE_DESC'].astype('category')  # Description of property use
df['BOROUGH_DESC'] = df['BOROUGH_DESC'].astype('category')  # Description of borough
df['FLOOR'] = df['FLOOR'].astype('category') # The floor of the building where the incident took place


# Numerical data type
# Columns representing numerical values are set to an integer data type.
# The 'errors' parameter is set to 'ignore' to avoid errors when conversion isn't possible,
# which can happen if there are missing or non-numeric values.
df['UNITS_ONSCENE'] = df['UNITS_ONSCENE'].astype('int64', errors='ignore')
df['RESPONSE_TIME'] = df['RESPONSE_TIME'].astype('int64', errors='ignore')
df['RESPONSE_TIME_MINUTES'] = df['RESPONSE_TIME_MINUTES'].astype('float64', errors='ignore')
df['LATITUDE'] = df['LATITUDE'].astype('float64', errors='ignore')
df['LONGITUDE'] = df['LONGITUDE'].astype('float64', errors='ignore')

# Datetime data types
# Columns representing dates and times are converted to the 'datetime64' data type
# to enable date and time operations and analyses.
df['INCIDENT_DATE_TIME'] = pd.to_datetime(df['INCIDENT_DATE_TIME'])  # Incident date and time
df['YEAR'] = df['INCIDENT_DATE_TIME'].dt.year
df['INCIDENT_MONTH'] = df['INCIDENT_DATE_TIME'].dt.month
df['INCIDENT_DAY_OF_WEEK'] = df['INCIDENT_DATE_TIME'].dt.day
df['INCIDENT_HOUR'] = df['INCIDENT_DATE_TIME'].dt.hour

df = df.drop('INCIDENT_DATE_TIME', axis=1)
df = df.drop('UNITS_ONSCENE_log', axis=1)

# Features and target variables
X = df.drop(['UNITS_ONSCENE'], axis=1)
y_original = df['UNITS_ONSCENE']

# Splitting data into training and temporary (validation + test) subsets
X_train, X_temp, y_train_orig, y_temp_orig = train_test_split(X, y_original, test_size=0.4, random_state=42)
X_val, X_test, y_val_orig, y_test_orig = train_test_split(X_temp, y_temp_orig, test_size=0.5, random_state=42)

# Replace inf/-inf with NaN
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)

# Option 1: Remove rows with NaN values
X_train.dropna(inplace=True)

# If you choose to drop rows, you need to align y_train_orig and y_train_log
y_train_orig = y_train_orig[X_train.index]

# Drop rows with NaNs:
X_val.dropna(inplace=True)
X_test.dropna(inplace=True)

# Align y_val_orig, y_test_orig with the new X_val and X_test
y_val_orig = y_val_orig[X_val.index]
y_test_orig = y_test_orig[X_test.index]

# Load model
best_lgbm_model = joblib.load('best_lgbm_model.pkl')

# Define categorical columns
categorical_columns = ['IM_INCIDENT_KEY', 'FIRE_BOX', 'STREET_HIGHWAY', 'ZIP_CODE', 'INCIDENT_TYPE_DESC', 'PROPERTY_USE_DESC', 'BOROUGH_DESC', 'FLOOR']

# Initialize LabelEncoder
encoder = LabelEncoder()

# Replace categorical columns in the dataframe with encoded values
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

def evauation_model(pred, y_val):
    score_MSE = round(mean_squared_error(pred, y_val), 2)
    score_MAE = round(mean_absolute_error(pred, y_val), 2)
    score_r2score = round(r2_score(pred, y_val), 2)
    return score_MSE, score_MAE, score_r2score


# Header for your app
st.header("Incident Unit Prediction App")
st.text_input("Enter your Name: ", key="name")

st.subheader("Please enter the incident details for prediction")


# Function to create a select box with searchable options
def create_selectbox(label, options):
    return st.selectbox(label, options, format_func=lambda x: x if x != "" else "None")


# Create searchable input fields
input_FIRE_BOX = create_selectbox("Fire Box", df['FIRE_BOX'].unique())
input_STREET_HIGHWAY = create_selectbox("Street/Highway", df['STREET_HIGHWAY'].unique())
input_ZIP_CODE = create_selectbox("Zip Code", df['ZIP_CODE'].unique())
input_INCIDENT_TYPE_DESC = create_selectbox("Incident Type Description", df['INCIDENT_TYPE_DESC'].unique())
input_PROPERTY_USE_DESC = create_selectbox("Property Use Description", df['PROPERTY_USE_DESC'].unique())
input_BOROUGH_DESC = create_selectbox("Borough Description", df['BOROUGH_DESC'].unique())
input_INCIDENT_DAY_OF_WEEK = create_selectbox("Incident Day of Week", df['INCIDENT_DAY_OF_WEEK'].unique())
input_INCIDENT_HOUR = create_selectbox("Incident Hour", df['INCIDENT_HOUR'].unique())
input_INCIDENT_MONTH = create_selectbox("Incident Month", df['INCIDENT_MONTH'].unique())
input_YEAR = create_selectbox("Year", df['YEAR'].unique())

# Other input fields
input_IM_INCIDENT_KEY = st.text_input("IM Incident Key")
input_FLOOR = st.text_input("Floor")
input_RESPONSE_TIME = st.number_input("Response Time", min_value=0)
input_RESPONSE_TIME_MINUTES = input_RESPONSE_TIME / 60
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
            # Check if the value is in the encoder's classes
            if encoded_data[feature] in encoder.classes_:
                encoded_value = encoder.transform([encoded_data[feature]])
                encoded_data[feature] = encoded_value[0]
            else:
                # Handle unseen labels (you can assign a default value or exclude them)
                encoded_data[feature] = -1  # Example: using -1 for unseen labels
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

    score_MSE, score_MAE, score_r2score = evauation_model(prediction, y_test_orig)
    print(score_MSE, score_MAE, score_r2score)

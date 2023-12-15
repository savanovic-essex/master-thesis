import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evauation_model(pred, y_val):
    score_MSE = round(mean_squared_error(pred, y_val), 2)
    score_MAE = round(mean_absolute_error(pred, y_val), 2)
    score_r2score = round(r2_score(pred, y_val), 2)
    return score_MSE, score_MAE, score_r2score


df = pd.read_csv("incidents.csv")

# Set appropriate data types for various columns

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

# Define categorical columns
categorical_columns = ['IM_INCIDENT_KEY', 'FIRE_BOX', 'STREET_HIGHWAY', 'ZIP_CODE', 'INCIDENT_TYPE_DESC', 'PROPERTY_USE_DESC', 'BOROUGH_DESC', 'FLOOR']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Replace categorical columns in the dataframe with encoded values
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

np.save('classes.npy', label_encoder.classes_)

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

# load model
best_lgbm_model = joblib.load('best_lgbm_model.pkl')
pred = best_lgbm_model.predict(X_test)
score_MSE, score_MAE, score_r2score = evauation_model(pred, y_test_orig)
print(score_MSE, score_MAE, score_r2score)


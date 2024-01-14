# Master Thesis Backend

## Description
This repository hosts the backend for my Master's thesis project, titled "Optimising Fire Emergency Response in Urban Environments: A Machine Learning Approach to Predict Unit Deployment".
It features a machine learning model that predicts the required number of firefighter units needed on the scene.
The application exposes an API endpoint which accepts specific arguments and feeds them into the ML model to return the predicted amount of units needed on the scene.

## Folder & File Structure
   ```
   .
├── Merged_Coordidnates_Incidents_Responded_to_by_Fire_Companies.csv
├── Procfile
├── README.md
├── app.py
├── classes.npy
├── lgbm_model.pkl
├── master_thesis.ipynb
└── requirements.txt
   ```

## Technology Stack
- Flask: A lightweight WSGI web application framework used to create the API.
- LightGBM: A gradient boosting framework for machine learning to build the predictive model.
- Pickle: For loading the serialized Python object (the ML model).
- NumPy: Utilized for array operations and data handling.
- Flask-CORS: Handles Cross-Origin Resource Sharing (CORS), allowing the API to be accessible from web applications.
- Gunicorn: Serves as the HTTP server for the Flask application.
- Jupyter Notebook: Used for exploratory data analysis, model training, and testing.

## Installation
To set up this project locally:

1. Clone the repository to your local machine.
   ```
   git clone https://github.com/savanovic-essex/master-thesis-backend.git
   ```

2. Install the required dependencies:
   ```
   pip3 install -r requirements.txt
   ```

## Usage
After installation, run the application using:

```
python3 app.py
```

The application will start a Flask server and expose an endpoint for making predictions.
The endpoint will be available on this http://127.0.0.1:5000 URL.

## Functionality
The main file, containing the business logic is called `app.py`.

This is how it works:

1. **Imports and Initial Setup**: It imports necessary libraries like Flask, pickle, NumPy, and others. Flask creates the web server, pickle is used to load the machine learning model, NumPy for array operations, and Flask-CORS to handle cross-origin resource sharing.

2. **Flask App Initialisation**: Initialises a Flask app and configures CORS, making the API accessible to web applications.

3. **Label Encoder Loading**: Loads a LabelEncoder (saved in 'classes.npy') for converting categorical features to numeric, as machine learning models require numerical inputs.

4. **Feature Encoding Function**: Defines `encode_features`, which encodes given input data's categorical features using the loaded LabelEncoder.

5. **Model Loading**: Loads a pre-trained LightGBM model from 'lgbm_model.pkl' using pickle.

6. **API Endpoints**:
   - A root endpoint (`'/'`) that returns a simple string as a response.
   - A prediction endpoint (`'/predict'`), which accepts POST requests. It processes the input data, encodes it, makes a prediction using the LightGBM model, and returns the prediction.

7. **Running the App**: Finally, if the script is run directly, it starts the Flask app with debug mode enabled.

This setup allows the application to receive data via HTTP POST requests, process this data using the machine learning model, and return predictions as responses.

## Additional files
- This repository also serves as a host for the final ML model in my Master's thesis.
The .pkl file is located in the root and is called `lgbm_model.pkl`.

- Additionally, this repository hosts the final dataset, which was used for training the model.
The file is also located in the root folder and is called `Merged_Coordidnates_Incidents_Responded_to_by_Fire_Companies.csv`.

- Finally, it hosts the Jupyter Notebook, which was used for the whole project, including EDA, model training and testing, etc. The file is located in the root folder and is called `master_thesis.ipynb`. 

## Contact
For any queries or collaborations, feel free to contact me at [dsavanovic@yahoo.com].

## References
flask.palletsprojects.com. (n.d.) Welcome to Flask — Flask Documentation (3.0.x). Available from: https://flask.palletsprojects.com/en/3.0.x/ [Accessed 19 November 2023].

Google (2019) Google Colaboratory. Google.com. Available from: https://colab.research.google.com/ [Accessed 15 July 2023].

joblib.readthedocs.io. (n.d.) Joblib: running Python functions as pipeline jobs — joblib 1.3.2 documentation. Available from: https://joblib.readthedocs.io/en/stable/ [Accessed 10 November 2023].

lightgbm.readthedocs.io. (n.d.) Welcome to LightGBM’s documentation! — LightGBM 3.3.5 documentation. Available from: https://lightgbm.readthedocs.io/en/stable/ [Accessed 16 July 2023].

Numpy (2009) NumPy. Numpy.org. Available from: https://numpy.org/ [Accessed 07 July 2023].

scikit-learn (2019) scikit-learn: machine learning in Python. Scikit-learn.org. Available from: https://scikit-learn.org/stable/ [Accessed 07 July 2023].

shap.readthedocs.io. (n.d.) Welcome to the SHAP Documentation — SHAP latest documentation. Available from: https://shap.readthedocs.io/en/latest/ [Accessed 15 December 2023].
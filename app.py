# import os
# import pandas as pd
# from flask import Flask, render_template, request, send_file
# import pickle
# import gzip
# from feature_engineering import preprocess_and_engineer_features  # Import feature engineering

# app = Flask(__name__)

# # Path to the compressed model file (pickle_gb.pkl.gz)
# model_path = 'models/pickle_rf.pkl.gz'

# UPLOAD_FOLDER = os.path.abspath('uploads')
# PROCESSED_FOLDER = os.path.abspath('processed')

# # Ensure the directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# # Load the trained GradientBoosting model from the compressed pickle file
# try:
#     with gzip.open(model_path, 'rb') as file:
#         model = pickle.load(file)
#     print("Model loaded successfully")
# except Exception as e:
#     print(f"Error loading model: {e}")


# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return render_template('error.html', error_message="No file part.")
    
#     file = request.files['file']
    
#     if file and file.filename.endswith('.csv'):
#         print("CSV file detected, processing...")
#         try:
#             # Read the CSV file into a DataFrame
#             df = pd.read_csv(file)
#             original_df = df.copy()
#             print(f"CSV Data:\n{df.head()}")  # Check if the CSV is loaded correctly

#             # Ensure the necessary columns are present
#             required_columns = ['Date', 'Trip Nr', 'Ordertype', 'PlannedDateTime', 'ArrivedDateTime', 'PlannedDay', 'PlannedHour', 'CustomerID', 'RelationID', 'CarrierID', 'NumberOfOrders']
#             missing_columns = [col for col in required_columns if col not in df.columns]
#             if missing_columns:
#                 return render_template('error.html', error_message=f"Missing columns: {', '.join(missing_columns)}")

#             # Process the data
#             processed_data = preprocess_and_engineer_features(df)  # Feature engineering
#             print(f"Processed Data:\n{processed_data}")  # Print processed data

#             # Predict the probability of delay and add columns to the DataFrame
#             y_prob_holdout = model.predict_proba(processed_data)[:, 1]
#             df['Percentage'] = y_prob_holdout * 100
#             df['Status'] = (y_prob_holdout >= 0.5).astype(int)
#             df['Status'] = df['Status'].map({1: 'Delayed', 0: 'On Time'})

#             # Create a processed file name and path
#             processed_file_name = f"results_{file.filename}"
#             processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_file_name)

#             # Append the results to the original DataFrame and save to CSV
#             columns_to_append = df[['Percentage', 'Status']]
#             final_df = pd.concat([original_df, columns_to_append], axis=1)
#             final_df.to_csv(processed_file_path, index=False)

#             message = f"The final CSV file has been saved in the processed folder as: {processed_file_name}"

#             return render_template('results.html', file_name=processed_file_name, message=message)

#         except Exception as e:
#             print(f"Error during processing: {e}")
#             return render_template('error.html', error_message="An error occurred during processing.")
#     else:
#         return render_template('error.html', error_message="Invalid file format. Please upload a CSV file.")


# @app.route('/download/<filename>')
# def download_file(filename):
#     file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
#     print(f"Attempting to download file from: {file_path}")  # Debugging
#     if not os.path.exists(file_path):
#         return f"Error: File not found at {file_path}"
#     return send_file(file_path, as_attachment=True)


# if __name__ == '__main__':
#     app.run(debug=True)


# ########################################################################################################## 
#                            ### #Using the Pre processor pickle File####
# ##########################################################################################################

import os
import pandas as pd
from flask import Flask, render_template, request, send_file
import pickle
import gzip
import importlib

from feature_engineering import preprocess_and_engineer_features  # Import original feature engineering function

app = Flask(__name__)

# Path to the compressed model file (pickle_gb.pkl.gz)
model_path = 'models/pickle_rf.pkl.gz'

UPLOAD_FOLDER = os.path.abspath('uploads')
PROCESSED_FOLDER = os.path.abspath('processed')

# Ensure the directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Load the trained GradientBoosting model from the compressed pickle file
try:
    with gzip.open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# feature_engineering = importlib.import_module('feature_engineering')
# Load the feature engineering function under a new name
loaded_preprocessing_function = None
try:
    with open('models/preprocessing_pipeline.pkl', 'rb') as f:
        loaded_preprocessing_function = pickle.load(f)
    print("Preprocessing function loaded successfully as 'loaded_preprocessing_function'")
except Exception as e:
    print(f"Error loading preprocessing function: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('error.html', error_message="No file part.")
    
    file = request.files['file']
    
    if file and file.filename.endswith('.csv'):
        print("CSV file detected, processing...")
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)
            original_df = df.copy()
            print(f"CSV Data:\n{df.head()}")  # Check if the CSV is loaded correctly

            # Ensure the necessary columns are present
            required_columns = ['Date', 'Trip Nr', 'Ordertype', 'PlannedDateTime', 'ArrivedDateTime', 'PlannedDay', 'PlannedHour', 'CustomerID', 'RelationID', 'CarrierID', 'NumberOfOrders']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return render_template('error.html', error_message=f"Missing columns: {', '.join(missing_columns)}")

            # Process the data using the loaded preprocessing function
            processed_data = loaded_preprocessing_function(df)  # Feature engineering
            print(f"Processed Data:\n{processed_data}")  # Print processed data

            # Predict the probability of delay and add columns to the DataFrame
            y_prob_holdout = model.predict_proba(processed_data)[:, 1]
            df['Percentage'] = y_prob_holdout * 100
            df['Status'] = (y_prob_holdout >= 0.5).astype(int)
            df['Status'] = df['Status'].map({1: 'Delayed', 0: 'On Time'})

            # Create a processed file name and path
            processed_file_name = f"results_{file.filename}"
            processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_file_name)

            # Append the results to the original DataFrame and save to CSV
            columns_to_append = df[['Percentage', 'Status']]
            final_df = pd.concat([original_df, columns_to_append], axis=1)
            final_df.to_csv(processed_file_path, index=False)

            message = f"The final CSV file has been saved in the processed folder as: {processed_file_name}"

            return render_template('results.html', file_name=processed_file_name, message=message)

        except Exception as e:
            print(f"Error during processing: {e}")
            return render_template('error.html', error_message="An error occurred during processing.")
    else:
        return render_template('error.html', error_message="Invalid file format. Please upload a CSV file.")


@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    print(f"Attempting to download file from: {file_path}")  # Debugging
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)

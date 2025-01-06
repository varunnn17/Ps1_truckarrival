# # # app.py

# # from flask import Flask, request, jsonify, render_template
# # import pickle
# # import numpy as np
# # import pandas as pd
# # from feature_engineering import feature_engineering

# # # Load the trained model
# # model_path = 'model.pkl'
# # with open(model_path, 'rb') as file:
# #     model = pickle.load(file)

# # app = Flask(__name__)

# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         # # Step 1: Extracting form data for each field
# #         # client = request.form.get('Client')
# #         # carrier = request.form.get('Carrier')
# #         # trip_mls = request.form.get('Trip.(MLS)')
# #         # order_count = request.form.get('Order Count')
# #         # confirmed_pallets = request.form.get('Confirmed Pallets')
# #         # load_sequence_count = request.form.get('LoadSequence Count')
# #         # buffer_assign = request.form.get('Buffer Assign')
# #         # planned_arrival = request.form.get('Planned Arrival')
# #         # picked_pallets = request.form.get('PickedPallets')
# #         # latest_pick = request.form.get('latestPick')

# #         # # Step 2: Create a dictionary with these values
# #         # data = {
# #         #     'Client': client,
# #         #     'Carrier': carrier,
# #         #     'Trip.(MLS)': trip_mls,
# #         #     'Order Count': order_count,
# #         #     'Confirmed Pallets': confirmed_pallets,
# #         #     'LoadSequence Count': load_sequence_count,
# #         #     'Buffer Assign': buffer_assign,
# #         #     'Planned Arrival': planned_arrival,
# #         #     'PickedPallets': picked_pallets,
# #         #     'latestPick': latest_pick
# #         # }
        
# #         data = request.form.to_dict()

# #         # Debugging: Print the input data to check structure
# #         print("Input Data: ", data)

# #         # Step 2: Convert the dictionary to a pandas DataFrame
# #         input_df = pd.DataFrame([data])  # This assumes each request contains a single data row
        
# #         # Debugging: Print the DataFrame to inspect it
# #         print("Input DataFrame: ", input_df)

# #         # Step 3: Check if the input data is valid
# #         if input_df.isnull().values.any():
# #             return jsonify({'error': 'Missing values in input data'}), 400
        
# #         # Step 4: Preprocess the data with feature engineering
# #         engineered_df = feature_engineering(input_df)
        
# #         # Debugging: Print the engineered DataFrame
# #         print("Engineered DataFrame: ", engineered_df)

# #         # Ensure engineered_df has the expected shape and columns
# #         if engineered_df.empty or engineered_df.shape[1] == 0:
# #             return jsonify({'error': 'Feature engineering failed, empty dataframe'}), 400
        
# #         # Step 5: Make a prediction using your model
# #         prediction = model.predict(engineered_df)  # Assuming model.predict() is used
        
# #         # Step 6: Return the prediction as a JSON response
# #         return jsonify({'prediction': prediction.tolist()})
    
# #     except Exception as e:
# #         # Step 7: Handle exceptions and return error if any
# #         return jsonify({'error': str(e)}), 400




# # if __name__ == "__main__":
# #     app.run(debug=True)

# from flask import Flask, render_template, request, url_for, redirect
# import pickle
# import gzip
# from feature_engineering import preprocess_and_engineer

# # Path to the compressed model file (model_zip.pkl.gz)
# model_path = 'model_zip.pkl.gz'

# # Load the trained VotingClassifier model from the compressed pickle file
# try:
#     with gzip.open(model_path, 'rb') as file:
#         model = pickle.load(file)
#     print(f"Model loaded successfully from: {model_path}")
# except Exception as e:
#     print(f"An error occurred while loading the model: {e}")

# app = Flask(__name__)

# # Route for the home page
# @app.route('/')
# def home():
#     return render_template('home.html')

# # Route for the prediction form
# @app.route('/index')
# def index():
#     return render_template('index.html')

# # Route to handle the form submission
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Capture form data
#     prediction = None
#     delay_percentage = None
#     status = None
#     if request.method == 'POST':
#         try:
#             data = {
#                     'Date': request.form.get('Date'),
#                     'Trip Nr': int(request.form.get('Trip Nr')),
#                     'Ordertype': request.form.get('Ordertype'),
#                     'PlannedDateTime': request.form.get('PlannedDateTime'),
#                     'PlannedDay': int(request.form.get('PlannedDay')),
#                     'PlannedHour': int(request.form.get('PlannedHour')),
#                     'CustomerID': int(request.form.get('CustomerID')),
#                     'RelationID': int(request.form.get('RelationID')),
#                     'CarrierID': int(request.form.get('CarrierID')),
#                     'NumberOfOrders': int(request.form.get('NumberOfOrders'))
#                 }
#             processed_data = preprocess_and_engineer(data)
#             y_prob_holdout = model.predict_proba(processed_data)[:, 1]
#             delay_percentage = y_prob_holdout[0] * 100
#             y_pred_holdout = (y_prob_holdout >= 0.5).astype(int)
#             status = "Delayed" if y_pred_holdout[0] == 1 else "On Time"
#             prediction = f"{delay_percentage:.2f}% potential of truck getting delayed"
#         except Exception as e:
#             print(f"Error during prediction: {e}", flush=True)
#             status = "Error: An unexpected error occurred."

#     return redirect(url_for('results', prediction=prediction, status=status))

# @app.route('/results', methods=['GET'])
# def results():
#     # Render a separate page for the prediction results
#     prediction = request.args.get('prediction')
#     status = request.args.get('status')
#     return render_template('results.html', prediction=prediction, status=status)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, url_for, redirect
import pickle
import gzip
from feature_engineering import preprocess_and_engineer

# Path to the compressed model file (pickle_gb.pkl.gz)
model_path = r"C:\Users\gopalv\Desktop\Truckarrival-Render\pickle_gb.pkl.gz"

# Load the trained GradientBoosting model from the compressed pickle file
try:
    with gzip.open(model_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for the prediction form
@app.route('/index')
def index():
    return render_template('index.html')

# Route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Capture form data
    prediction = None
    delay_percentage = None
    status = None
    if request.method == 'POST':
        try:
            data = {
                    'Date': request.form.get('Date'),
                    'Trip Nr': int(request.form.get('Trip Nr')),
                    'Ordertype': request.form.get('Ordertype'),
                    'PlannedDateTime': request.form.get('PlannedDateTime'),
                    'PlannedDay': int(request.form.get('PlannedDay')),
                    'PlannedHour': int(request.form.get('PlannedHour')),
                    'CustomerID': int(request.form.get('CustomerID')),
                    'RelationID': int(request.form.get('RelationID')),
                    'CarrierID': int(request.form.get('CarrierID')),
                    'NumberOfOrders': int(request.form.get('NumberOfOrders'))
                }
            processed_data = preprocess_and_engineer(data)
            y_prob_holdout = model.predict_proba(processed_data)[:, 1]
            delay_percentage = y_prob_holdout[0] * 100
            y_pred_holdout = (y_prob_holdout >= 0.5).astype(int)
            status = "Delayed" if y_pred_holdout[0] == 1 else "On Time"
            prediction = f"{delay_percentage:.2f}% potential of truck getting delayed"
        except Exception as e:
            print(f"Error during prediction: {e}", flush=True)
            status = "Error: An unexpected error occurred."

    return redirect(url_for('results', prediction=prediction, status=status))

@app.route('/results', methods=['GET'])
def results():
    # Render a separate page for the prediction results
    prediction = request.args.get('prediction')
    status = request.args.get('status')
    return render_template('results.html', prediction=prediction, status=status)

if __name__ == '__main__':
    app.run(debug=True)

# model.py
import gzip
import pickle
# from joblib import load

MODEL_PATH = 'models\pickle_rf.pkl.gz.'
# Load the trained model from file
def load_model():
# Load the compressed model
    with gzip.open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model
    

# Make predictions using the preprocessed data
def predict_delay(model, data):
    """Predict the delay probability using the trained model."""
    return model.predict_proba(data)[:, 1] * 100


#     with gzip.open(model_path, 'rb') as file:
#         model = pickle.load(file)
#     print("Model loaded successfully")
# except Exception as e:
#     print(f"Error loading model: {e}")
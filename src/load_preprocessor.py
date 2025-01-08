import pickle

PREPROCESSOR_PATH = 'models/preprocessor.pkl'
def load_preprocessor(filename='models/preprocessor.pkl'):
    with open(filename, 'rb') as f:
        preprocessor = pickle.load(f)
    return preprocessor


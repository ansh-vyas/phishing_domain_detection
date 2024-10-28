import pickle
import logging
import numpy as np

MODEL_PATH = 'model.pkl'  # Path to the trained model file

def load_model(file_path=MODEL_PATH):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    logging.info("Model loaded from %s", file_path)
    return model

def predict(model, X):
    prediction = model.predict(X)
    logging.info("Prediction made.")
    return prediction

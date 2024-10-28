import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from .data_preprocessing import load_data, preprocess_data

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logging.info("Model evaluation:\n%s", report)
    return report

def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    logging.info("Model saved to %s", file_path)

# src/model_training.py

import logging
import os
import joblib
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logging_config import initialize_logging

# Initialize logging
initialize_logging()

# Load configuration from YAML file
config_path = os.path.join("config\\config.yaml")
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Paths for data
processed_data_path = config["paths"]["processed_data"]
model_output_path = config["paths"]["model_output"]

# Load processed dataset
def load_data():
    try:
        data = pd.read_csv(processed_data_path)
        logging.info("Data loaded successfully for model training.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Split the data into training and testing sets
def split_data(data):
    X = data.drop('phishing', axis=1)
    y = data['phishing']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["train_test_split"]["test_size"], random_state=config["train_test_split"]["random_state"])
    logging.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=config["model_params"]["n_estimators"],
        max_depth=config["model_params"]["max_depth"],
        random_state=config["model_params"]["random_state"]
    )
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    logging.info(f"Model Evaluation - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    return accuracy, precision, recall, f1

# Save the model
def save_model(model):
    os.makedirs(model_output_path, exist_ok=True)
    model_file = os.path.join(model_output_path, "trained_model.pkl")
    joblib.dump(model, model_file)
    logging.info(f"Model saved to {model_file}")

# Main execution
def main():
    try:
        data = load_data()
        X_train, X_test, y_train, y_test = split_data(data)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model)
        logging.info("Model training pipeline completed successfully.")
    except Exception as e:
        logging.info(f"Error in model training pipeline: {e}")

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os

# Initialize logging
from src.logging_config import initialize_logging

initialize_logging()

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            logging.error("The provided CSV file is empty.")
            raise ValueError("The CSV file is empty.")
        
        logging.info("Data loaded successfully from %s", file_path)
        return data
    except pd.errors.EmptyDataError:
        logging.error("No data found in the CSV file.")
        raise
    except pd.errors.ParserError:
        logging.error("Parsing error: The file may not be formatted as a valid CSV.")
        raise
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise

def preprocess_data(data):
    # Define feature columns based on your data's structure
    url_features = [
        'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
        'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url',
        'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url', 'qty_comma_url',
        'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url', 'qty_dollar_url',
        'qty_percent_url', 'qty_tld_url', 'length_url', 'email_in_url'
    ]
    
    domain_features = [
        'qty_dot_domain', 'qty_hyphen_domain', 'qty_underline_domain', 'qty_slash_domain',
        'qty_questionmark_domain', 'qty_equal_domain', 'qty_at_domain', 'qty_and_domain',
        'qty_exclamation_domain', 'qty_space_domain', 'qty_tilde_domain', 'qty_comma_domain',
        'qty_plus_domain', 'qty_asterisk_domain', 'qty_hashtag_domain', 'qty_dollar_domain',
        'qty_percent_domain', 'qty_vowels_domain', 'domain_length', 'domain_in_ip',
        'server_client_domain'
    ]
    
    resolving_features = [
        'time_response', 'domain_spf', 'asn_ip', 'time_domain_activation', 'time_domain_expiration',
        'qty_ip_resolved', 'qty_nameservers', 'qty_mx_servers', 'ttl_hostname', 'tls_ssl_certificate',
        'qty_redirects', 'url_google_index', 'domain_google_index', 'url_shortened'
    ]

    feature_columns = url_features + domain_features + resolving_features

    # Check if required columns are in the data
    missing_columns = [col for col in feature_columns if col not in data.columns]
    if missing_columns:
        logging.error("Missing columns in the data: %s", missing_columns)
        raise ValueError(f"Missing columns: {missing_columns}")
    
    # Select features and target variable
    X = data[feature_columns]
    y = data['phishing']  # Adjust if target column name is different

    # Handle missing values
    X.fillna(0, inplace=True)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Feature scaling applied.")

    # Combine scaled features and target into a single DataFrame
    processed_data = pd.DataFrame(X_scaled, columns=feature_columns)
    processed_data['phishing'] = y.reset_index(drop=True)

    # Save processed data to a single CSV file
    os.makedirs('data', exist_ok=True)
    processed_data.to_csv('data/processed_data.csv', index=False)
    logging.info("Processed data saved to data/processed_data.csv.")

    return processed_data

# Paths for input and output files
input_file_path = 'data/dataset_full.csv'

# Load data, preprocess, and save
try:
    data = load_data(input_file_path)
    processed_data = preprocess_data(data)
    logging.info("Data preprocessing completed successfully.")
except Exception as e:
    logging.error("An error occurred during preprocessing: %s", e)

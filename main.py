import os
import pandas as pd
from src.utils.fetch_bitcoin_data import fetch_latest_bitcoin_data
from src.data_preprocessing.preprocess_data import preprocess_bitcoin_data
from src.eda.bitcoin_eda import perform_eda  # Import the EDA function
from src.feature_engineering.feature_engineering import feature_engineering  # Import the feature engineering function

def main():
    # Define paths
    raw_data_path = os.path.join('data', 'raw', 'bitcoin_data.csv')
    processed_data_path = os.path.join('data', 'processed', 'processed_bitcoin_data.csv')
    engineered_data_path = os.path.join('data', 'processed', 'engineered_bitcoin_data.csv')

    # Fetch Bitcoin data
    print("Fetching Bitcoin data...")
    fetch_latest_bitcoin_data('402db71a-cbac-4ff7-8120-2053b050d2ae', raw_data_path)
    print("Data fetched successfully!")

    # Preprocess the fetched data
    print("Preprocessing data...")
    preprocess_bitcoin_data(raw_data_path, processed_data_path)
    print("Data preprocessed successfully!")

    # Feature Engineering on the processed data
    print("Performing Feature Engineering...")
    df = pd.read_csv(processed_data_path)
    df = feature_engineering(df)
    df.to_csv(engineered_data_path, index=False)
    print("Feature Engineering completed successfully!")

    # Perform EDA on the engineered data
    print("Performing Exploratory Data Analysis (EDA)...")
    perform_eda(engineered_data_path)

if __name__ == "__main__":
    main()

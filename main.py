import os
import pandas as pd
from src.utils.fetch_bitcoin_data import fetch_latest_bitcoin_data
from src.data_preprocessing.preprocess_data import preprocess_bitcoin_data
from src.eda.bitcoin_eda import perform_eda  # Import the EDA function
from src.feature_engineering.feature_engineering import feature_engineering  # Import the feature engineering function
from src.models.train import load_data, train_models, evaluate_models
from src.data_preprocessing.clean_data import clean_data

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

    # Load the preprocessed data into a DataFrame
    df = pd.read_csv(processed_data_path)

    # Clean the preprocessed data
    print("Cleaning data...")
    df = clean_data(df)
    df.to_csv(processed_data_path, index=False)  # Overwrite the preprocessed data with cleaned data
    print("Data cleaned successfully!")

    # Feature Engineering on the cleaned data
    print("Performing Feature Engineering...")
    df = feature_engineering(df)
    df.to_csv(engineered_data_path, index=False)
    print("Feature Engineering completed successfully!")

    # Perform EDA on the engineered data
    print("Performing Exploratory Data Analysis (EDA)...")
    perform_eda(engineered_data_path)

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_data(engineered_data_path)

    # Train models
    trained_models = train_models(X_train, y_train)

    # Evaluate models
    evaluate_models(trained_models, X_test, y_test)

if __name__ == "__main__":
    main()

import os
from src.utils.fetch_bitcoin_data import fetch_bitcoin_data
from src.data_preprocessing.preprocess_data import preprocess_bitcoin_data

def main():
    # Define paths
    raw_data_path = os.path.join('data', 'raw', 'bitcoin_data.csv')
    processed_data_path = os.path.join('data', 'processed', 'processed_bitcoin_data.csv')

    # Fetch Bitcoin data
    print("Fetching Bitcoin data...")
    fetch_bitcoin_data('402db71a-cbac-4ff7-8120-2053b050d2ae', raw_data_path)
    print("Data fetched successfully!")

    # Preprocess the fetched data
    print("Preprocessing data...")
    preprocess_bitcoin_data(raw_data_path, processed_data_path)
    print("Data preprocessed successfully!")

if __name__ == "__main__":
    main()

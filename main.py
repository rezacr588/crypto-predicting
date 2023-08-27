import os
import pandas as pd
from src.utils.fetch_bitcoin_data import fetch_latest_bitcoin_data
from src.data_preprocessing.preprocess_data import BitcoinPreprocessor
from src.eda.bitcoin_eda import perform_eda
from src.models.train import BitcoinModel

# Configuration
RAW_DATA_PATH = os.path.join('data', 'raw', 'bitcoin_data.csv')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'processed_bitcoin_data.csv')
API_KEY = '402db71a-cbac-4ff7-8120-2053b050d2ae'

def fetch_data():
    print("Fetching Bitcoin data...")
    fetch_latest_bitcoin_data(API_KEY, RAW_DATA_PATH)
    print("Data fetched successfully!")

def preprocess():
    print("Preprocessing data...")
    preprocessor = BitcoinPreprocessor(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    preprocessor.preprocess()
    print("Data preprocessed successfully!")

def perform_analysis():
    print("Performing Exploratory Data Analysis (EDA)...")
    perform_eda(PROCESSED_DATA_PATH)

def train_and_evaluate():
    model = BitcoinModel(PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test = model.load_data()
    X_train, X_test = model.scale_features(X_train, X_test)
    model.train(X_train, y_train)
    
    mse = model.evaluate(X_test, y_test)
    print(f"Ensemble MSE: {mse:.2f}")
    return model

def predict(model: BitcoinModel):
    df = pd.read_csv(PROCESSED_DATA_PATH)
    predictions = model.predict(df.drop(columns=['price']))
    for i, pred in enumerate(predictions['Holt-Winters']):
        print(f"Ensemble predicted Bitcoin price for day {i+1}: ${pred:.2f}")

def main():
    fetch_data()
    preprocess()
    predict(train_and_evaluate())

if __name__ == "__main__":
    main()

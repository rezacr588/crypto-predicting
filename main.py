import os
import pandas as pd
from src.utils.fetch_bitcoin_data import fetch_latest_bitcoin_data
from src.data_preprocessing.preprocess_data import preprocess_bitcoin_data
from src.eda.bitcoin_eda import perform_eda
from src.feature_engineering.feature_engineering import feature_engineering
from src.models.train import load_data, train_models, evaluate_models, ensemble_predictions, scale_features, predict_next_day
from sklearn.metrics import mean_squared_error
from src.data_preprocessing.clean_data import clean_data

# Configuration
RAW_DATA_PATH = os.path.join('data', 'raw', 'bitcoin_data.csv')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'processed_bitcoin_data.csv')
ENGINEERED_DATA_PATH = os.path.join('data', 'processed', 'engineered_bitcoin_data.csv')
API_KEY = '402db71a-cbac-4ff7-8120-2053b050d2ae'

def fetch_data():
    print("Fetching Bitcoin data...")
    fetch_latest_bitcoin_data(API_KEY, RAW_DATA_PATH)
    print("Data fetched successfully!")

def preprocess():
    print("Preprocessing data...")
    preprocess_bitcoin_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print("Data preprocessed successfully!")

def clean():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print("Cleaning data...")
    df = clean_data(df)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Data cleaned successfully!")

def engineer_features():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print("Performing Feature Engineering...")
    df = feature_engineering(df)
    df.to_csv(ENGINEERED_DATA_PATH, index=False)
    print("Feature Engineering completed successfully!")

def perform_analysis():
    print("Performing Exploratory Data Analysis (EDA)...")
    perform_eda(ENGINEERED_DATA_PATH)

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_data(ENGINEERED_DATA_PATH)
    X_train, X_test = scale_features(X_train, X_test)
    trained_models = train_models(X_train, y_train, incremental=True)
    model_predictions = evaluate_models(trained_models, X_train ,X_test)
    ensemble_pred = ensemble_predictions(model_predictions)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    print(f"Ensemble MSE: {ensemble_mse:.2f}")
    return trained_models

def predict(trained_models):
    df = pd.read_csv(ENGINEERED_DATA_PATH)
    prediction = predict_next_day(trained_models, df)
    for i, pred in enumerate(prediction):
        print(f"Ensemble predicted Bitcoin price for day {i+1}: ${pred:.2f}")

def main():
    fetch_data()
    preprocess()
    clean()
    engineer_features()
    perform_analysis()
    trained_models = train_and_evaluate()
    predict(trained_models)

if __name__ == "__main__":
    main()

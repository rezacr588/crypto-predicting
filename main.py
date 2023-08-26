import os
import pandas as pd
from src.utils.fetch_bitcoin_data import fetch_latest_bitcoin_data
from src.data_preprocessing.preprocess_data import preprocess_bitcoin_data
from src.eda.bitcoin_eda import perform_eda
from src.feature_engineering.feature_engineering import feature_engineering
from src.models.train import load_data, train_models, evaluate_models, ensemble_predictions, scale_features, predict_next_day
from sklearn.metrics import mean_squared_error
from src.data_preprocessing.clean_data import clean_data

def check_datetime_columns(df):
    datetime_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_value = df[col].iloc[0]
            try:
                pd.to_datetime(sample_value)
                datetime_cols.append(col)
            except:
                continue
    return datetime_cols

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

    # Check for datetime columns
    datetime_columns = check_datetime_columns(df)
    if datetime_columns:
        print(f"Columns with datetime strings: {datetime_columns}")
    else:
        print("No columns with datetime strings found.")

    # Perform EDA on the engineered data
    print("Performing Exploratory Data Analysis (EDA)...")
    perform_eda(engineered_data_path)

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_data(engineered_data_path)
    
    # Scale the features
    X_train, X_test = scale_features(X_train, X_test)
    
    trained_models = train_models(X_train, y_train, incremental=True)
    
    # Get predictions from each model
    model_predictions = evaluate_models(trained_models, X_test)
    
    # Ensemble the predictions
    ensemble_pred = ensemble_predictions(model_predictions)
    
    # Evaluate the ensemble
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    print(f"Ensemble MSE: {ensemble_mse:.2f}")
    
    # Predict the next day's Bitcoin price using all past data
    prediction = predict_next_day(trained_models, df)
    for i, pred in enumerate(prediction):
        print(f"Ensemble predicted Bitcoin price for day {i+1}: ${pred:.2f}")

if __name__ == "__main__":
    main()

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def load_data(processed_data_path):
    """
    Load and preprocess the data for training.

    Parameters:
    - processed_data_path (str): Path to the processed data CSV file.

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing data.
    """
    data = pd.read_csv(processed_data_path)
    
    # Convert the 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])
    
    # Extract year, month, day, and hour as separate features
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['hour'] = data['date'].dt.hour
    
    # Drop the original 'date' column
    data = data.drop(columns=['date', 'last_updated'])  # Also dropping 'last_updated' if it's a similar datetime column
    
    X = data.drop(columns=['price'])
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """
    Always train the regression models and save them.

    Parameters:
    - X_train, y_train: Training data.

    Returns:
    - dict: Trained models.
    """
    model_paths = {
        'Linear Regression': 'models/linear_regression_model.pkl',
        'Random Forest': 'models/random_forest_model.pkl'
    }

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print(f"{name} trained successfully!")

        # Save the trained model
        with open(model_paths[name], 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"{name} model saved to {model_paths[name]}!")

    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data.

    Parameters:
    - models (dict): Trained models.
    - X_test, y_test: Testing data.

    Returns:
    None
    """
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name} MSE: {mse:.2f}")

if __name__ == "__main__":
    processed_data_path = '../data/processed/processed_bitcoin_data.csv'
    X_train, X_test, y_train, y_test = load_data(processed_data_path)
    trained_models = train_models(X_train, y_train)
    evaluate_models(trained_models, X_test, y_test)

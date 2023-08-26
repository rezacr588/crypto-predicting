import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler  
import numpy as np

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

def train_models(X_train, y_train, incremental=False):
    """
    Train the regression models and save them.

    Parameters:
    - X_train, y_train: Training data.
    - incremental (bool): If True, use incremental learning for regression.

    Returns:
    - dict: Trained models.
    """
    model_paths = {
        'Linear Regression': 'models/linear_regression_model.pkl',
        'Random Forest': 'models/random_forest_model.pkl'
    }

    models = {
        'Linear Regression': SGDRegressor(learning_rate='constant', eta0=0.01, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        if incremental and name == 'Linear Regression':
            # Split the training data into batches for incremental learning
            batch_size = 100  # You can adjust this value
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]  # Use numpy slicing
                y_batch = y_train.iloc[i:i+batch_size]
                model.partial_fit(X_batch, y_batch)
        else:
            model.fit(X_train, y_train)
        print(f"{name} trained successfully!")

        # Save the trained model
        with open(model_paths[name], 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"{name} model saved to {model_paths[name]}!")

    return models

def evaluate_models(models, X_test):
    """
    Evaluate trained models on test data and return predictions.

    Parameters:
    - models (dict): Trained models.
    - X_test: Testing data.

    Returns:
    - dict: Predictions from each model.
    """
    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
    return predictions

def ensemble_predictions(predictions):
    """
    Ensemble predictions from multiple models by averaging.

    Parameters:
    - predictions (dict): Predictions from each model.

    Returns:
    - array: Averaged predictions.
    """
    # Ensure there are models to ensemble
    if len(predictions) == 0:
        raise ValueError("No predictions to ensemble.")
    
    # Convert predictions to numpy arrays and average them
    return np.mean([np.array(pred) for pred in predictions.values()], axis=0)

def scale_features(X_train, X_test):
    """
    Scale features using Standard Scaling.

    Parameters:
    - X_train: Training data.
    - X_test: Testing data.

    Returns:
    - X_train_scaled, X_test_scaled: Scaled training and testing data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    processed_data_path = '../data/processed/processed_bitcoin_data.csv'
    X_train, X_test, y_train, y_test = load_data(processed_data_path)
    
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

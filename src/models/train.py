import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler  
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def load_data(processed_data_path):
    """
    Load and preprocess the data for training.

    Parameters:
    - processed_data_path (str): Path to the processed data CSV file.

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing data.
    """
    data = pd.read_csv(processed_data_path)
    
    X = data.drop(columns=['price'])
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Global variable to store the column names of X_train
TRAINED_COLUMNS = None

def train_models(X_train, y_train, incremental=False):
    global TRAINED_COLUMNS
    TRAINED_COLUMNS = X_train.columns  # Store the column names of X_train

    model_paths = {
        'Random Forest': 'models/random_forest_model.pkl',
        'Holt-Winters': 'models/holt_winters_model.pkl'
    }

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Holt-Winters': ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        if name == 'Holt-Winters':
            fitted_model = model.fit(optimized=True)
            models[name] = fitted_model
        else:
            model.fit(X_train, y_train)
        print(f"{name} trained successfully!")

        with open(model_paths[name], 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"{name} model saved to {model_paths[name]}!")

    return models

def evaluate_models(trained_models, X_train, X_test):
    global TRAINED_COLUMNS

    # Ensure X_test has the same columns in the same order as the trained model
    X_test = X_test.reindex(columns=TRAINED_COLUMNS)

    model_predictions = {}
    for name, model in trained_models.items():
        if name == 'Holt-Winters':
            y_pred = model.forecast(steps=len(X_test))
        else:
            y_pred = model.predict(X_test)
        model_predictions[name] = y_pred

    return model_predictions
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
    Scale the features using StandardScaler.

    Parameters:
    - X_train, X_test: Training and testing data.

    Returns:
    - X_train_scaled, X_test_scaled: Scaled training and testing data.
    """
    # Exclude non-numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled

def predict_next_day(models, recent_data_processed):
    predictions = {}
    X = recent_data_processed.drop(columns=['price'])

    for model_name, model in models.items():
        if model_name == 'Holt-Winters':
            prediction = model.forecast(steps=len(X))
        else:
            prediction = model.predict(X)
        predictions[model_name] = prediction

    ensemble_prediction = np.mean(list(predictions.values()), axis=0)
    return ensemble_prediction

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

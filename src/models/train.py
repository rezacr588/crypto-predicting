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
        'Linear Regression': 'models/linear_regression_model.pkl',
        'Random Forest': 'models/random_forest_model.pkl'
    }

    models = {
        'Linear Regression': SGDRegressor(learning_rate='constant', eta0=0.01, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Convert 'last_updated' column to Unix timestamp if it exists
    if 'last_updated' in X_train.columns:
        X_train['last_updated'] = pd.to_datetime(X_train['last_updated']).astype(int) / 10**9

    for name, model in models.items():
        print(f"Training {name}...")
        if incremental and name == 'Linear Regression':
            batch_size = 100
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train.iloc[i:i+batch_size]
                y_batch = y_train.iloc[i:i+batch_size]
                
                # Ensure no string columns in X_batch
                numeric_X_batch = X_batch.select_dtypes(exclude=['object'])
                if len(numeric_X_batch.columns) != len(X_batch.columns):
                    raise ValueError(f"X_batch contains non-numeric columns: {set(X_batch.columns) - set(numeric_X_batch.columns)}")
                
                model.partial_fit(numeric_X_batch, y_batch)
        else:
            model.fit(X_train, y_train)
        print(f"{name} trained successfully!")

        with open(model_paths[name], 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"{name} model saved to {model_paths[name]}!")

    return models

def evaluate_models(trained_models, X_test):
    """
    Evaluate the trained models on the test data.

    Parameters:
    - trained_models (dict): Dictionary of trained models.
    - X_test: Test data.

    Returns:
    - dict: Predictions from each model.
    """
    global TRAINED_COLUMNS

    # Ensure X_test has the same columns in the same order as the trained model
    X_test = X_test.reindex(columns=TRAINED_COLUMNS)

    model_predictions = {}
    for name, model in trained_models.items():
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
    """
    Predict Bitcoin price for all the data collected.

    Parameters:
    - models (dict): Trained models.
    - recent_data_processed (DataFrame or numpy array): Processed data.

    Returns:
    - dict: Ensemble predicted price for all the data.
    """
    predictions = {}

    # Drop the 'price' column to get the features
    X = recent_data_processed.drop(columns=['price'])

    # Ensure that the features in X match the features the model was trained on
    # This can be done by checking the number of features in X and the number of features in the model's `coef_` attribute
    for model_name, model in models.items():
        if hasattr(model, 'coef_'):
            if X.shape[1] != len(model.coef_):
                raise ValueError(f"Feature mismatch for {model_name}. Model expects {len(model.coef_)} features but got {X.shape[1]}.")
        prediction = model.predict(X)
        predictions[model_name] = prediction

    # Average the predictions from all models
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

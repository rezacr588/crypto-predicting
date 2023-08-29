import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression  # Importing Linear Regression
from sklearn.preprocessing import StandardScaler
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
class BitcoinModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.TRAINED_COLUMNS = None

    def load_data(self):
        data = pd.read_csv(self.data_path)
        X = data.drop(columns=['price'])
        y = data['price']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def scale_features(self, X_train, X_test):
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        return X_train, X_test

    def train(self, X_train, y_train):
        self.TRAINED_COLUMNS = X_train.columns
        # Convert 'datetime' column to pandas datetime object and set as index
        y_train_with_datetime = pd.DataFrame(y_train).set_index(pd.to_datetime(X_train['datetime']))
        
        model_data = {
            'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), 'models/random_forest_model.pkl'),
            'Linear Regression': (LinearRegression(), 'models/linear_regression_model.pkl'),
            'Holt-Winters': (ExponentialSmoothing(y_train_with_datetime, trend='add', seasonal='add', seasonal_periods=12), 'models/holt_winters_model.pkl')
        }

        for name, (model, path) in model_data.items():
            print(f"Training {name}...")
            if name == 'Holt-Winters':
                model = model.fit(optimized=True)
            else:
                # Drop the 'datetime' column for other models
                model.fit(X_train.drop(columns=['datetime']), y_train)
            print(f"{name} trained successfully!")
            with open(path, 'wb') as model_file:
                pickle.dump(model, model_file)
            print(f"{name} model saved to {path}!")
            self.models[name] = model
            
    def predict(self, X):
        predictions = {}
        for name, model in self.models.items():
            if name == 'Holt-Winters':
                # Convert 'datetime' column to pandas datetime object and set as index
                X_with_datetime = X.set_index(pd.to_datetime(X['datetime']))
                predictions[name] = model.forecast(steps=len(X_with_datetime))
            else:
                # Drop the 'datetime' column for other models
                predictions[name] = model.predict(X.drop(columns=['datetime']))
        return predictions

    def ensemble_predictions(self, predictions):
        return np.mean([np.array(pred) for pred in predictions.values()], axis=0)

    def evaluate(self, X_test, y_test):
        model_predictions = self.predict(X_test)
        
        # Dictionary to store MSE for each model
        mse_results = {}
        
        # Calculate MSE for each model
        for model_name, predictions in model_predictions.items():
            mse_results[model_name] = mean_squared_error(y_test, predictions)
        
        # Calculate MSE for the ensemble
        ensemble_pred = self.ensemble_predictions(model_predictions)
        mse_results["Ensemble"] = mean_squared_error(y_test, ensemble_pred)
        
        return mse_results


if __name__ == "__main__":
    DATA_PATH = '../data/processed/processed_bitcoin_data.csv'
    model = BitcoinModel(DATA_PATH)
    
    X_train, X_test, y_train, y_test = model.load_data()
    X_train, X_test = model.scale_features(X_train, X_test)
    model.train(X_train, y_train)
    
    mse_values = model.evaluate(X_test, y_test)
    for model_name, mse in mse_values.items():
        print(f"{model_name} MSE: {mse:.2f}")

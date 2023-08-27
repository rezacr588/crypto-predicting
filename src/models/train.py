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
        
        y_train_with_datetime_index = pd.DataFrame(y_train).set_index(X_train['datetime'])
        model_data = {
            'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), 'models/random_forest_model.pkl'),
            'Holt-Winters': (ExponentialSmoothing(y_train_with_datetime_index, trend='add', seasonal='add', seasonal_periods=12), 'models/holt_winters_model.pkl'),
            'Linear Regression': (LinearRegression(), 'models/linear_regression_model.pkl')
        }

        for name, (model, path) in model_data.items():
            print(f"Training {name}...")
            if name != 'Holt-Winters':
                X_train_model = X_train.drop(columns=['datetime'], errors='ignore')
            else:
                X_train_model = X_train
            if name == 'Holt-Winters':
                model = model.fit(optimized=True)
            else:
                model.fit(X_train_model, y_train)
            print(f"{name} trained successfully!")
            with open(path, 'wb') as model_file:
                pickle.dump(model, model_file)
            print(f"{name} model saved to {path}!")
            self.models[name] = model

    def predict(self, X):
        predictions = {}
        for name, model in self.models.items():
            if name != 'Holt-Winters':
                X_model = X.drop(columns=['datetime'], errors='ignore')
            else:
                X_model = X
            if name == 'Holt-Winters':
                predictions[name] = model.forecast(steps=len(X_model))
            else:
                predictions[name] = model.predict(X_model)
        return predictions


    def ensemble_predictions(self, predictions):
        return np.mean([np.array(pred) for pred in predictions.values()], axis=0)

    def evaluate(self, X_test, y_test):
        model_predictions = self.predict(X_test)
        ensemble_pred = self.ensemble_predictions(model_predictions)
        return mean_squared_error(y_test, ensemble_pred)

if __name__ == "__main__":
    DATA_PATH = '../data/processed/processed_bitcoin_data.csv'
    model = BitcoinModel(DATA_PATH)
    
    X_train, X_test, y_train, y_test = model.load_data()
    X_train, X_test = model.scale_features(X_train, X_test)
    model.train(X_train, y_train)
    
    mse = model.evaluate(X_test, y_test)
    print(f"Ensemble MSE: {mse:.2f}")

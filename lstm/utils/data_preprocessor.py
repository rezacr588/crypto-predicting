# /bitcoin_forecaster/utils/data_preprocessor.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import os

def preprocess_data(sequence_length=10):
    # Load the data from the CSV file
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bitcoin_hourly_prices.csv")
    df = pd.read_csv(data_path)
    
    # Extract closing prices
    closing_prices = df['close'].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(closing_prices)

    # Create sequences
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=sequence_length, batch_size=1)
    
    return generator, scaler

if __name__ == "__main__":
    generator, scaler = preprocess_data()
    print("Data preprocessing completed!")

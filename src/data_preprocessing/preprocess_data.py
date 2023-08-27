import pandas as pd
import numpy as np

class BitcoinPreprocessor:
    def __init__(self, input_path, output_path):
        self.data = pd.read_csv(input_path)
        self.output_path = output_path

    def fill_missing_values(self):
        self.data.fillna(method='ffill', inplace=True)
        self.data['tvl'].fillna(0, inplace=True)

    def compute_moving_average(self, window=7):
        self.data['7_day_avg'] = self.data['price'].rolling(window=window).mean()
        self.data['30_day_avg'] = self.data['price'].rolling(window=30).mean()

    def convert_data_types(self):
        numerical_cols = ['price', 'volume_24h', 'volume_change_24h', 'percent_change_1h', 'percent_change_24h', 
                          'percent_change_7d', 'percent_change_30d', 'percent_change_60d', 'percent_change_90d', 
                          'market_cap', 'market_cap_dominance', 'fully_diluted_market_cap', 'tvl']
        for col in numerical_cols:
            self.data[col] = self.data[col].astype(float)
        self.data['last_updated'] = pd.to_datetime(self.data['last_updated'], errors='coerce')

    def extract_date_features(self):
        self.data['date'] = self.data['last_updated'].dt.date
        self.data['day'] = self.data['last_updated'].dt.day
        self.data['month'] = self.data['last_updated'].dt.month
        self.data['year'] = self.data['last_updated'].dt.year

    def feature_engineering(self):
        # Convert 'date' column to Unix timestamp
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date']).astype(int) / 10**9

        # Convert 'last_updated' column to Unix timestamp
        if 'last_updated' in self.data.columns:
            self.data['last_updated'] = pd.to_datetime(self.data['last_updated']).astype(int) / 10**9

        # Date and Time Features
        self.data['day_of_week'] = pd.to_datetime(self.data['date'], unit='s').dt.dayofweek
        self.data['month'] = pd.to_datetime(self.data['date'], unit='s').dt.month
        self.data['year'] = pd.to_datetime(self.data['date'], unit='s').dt.year
        self.data['is_weekend'] = self.data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        # Lagged Features
        for lag in range(1, 8):  # for one week
            self.data[f'price_lag_{lag}'] = self.data['price'].shift(lag)

        # Rolling Window Statistics
        self.data['rolling_mean_7'] = self.data['price'].rolling(window=7).mean()
        self.data['rolling_std_7'] = self.data['price'].rolling(window=7).std()
        self.data['rolling_min_7'] = self.data['price'].rolling(window=7).min()
        self.data['rolling_max_7'] = self.data['price'].rolling(window=7).max()

        # Differences and Returns
        self.data['daily_return'] = self.data['price'].pct_change()
        self.data['log_return'] = np.log(self.data['price']).diff()

        # Exponential Weighted Features
        self.data['ewm_7'] = self.data['price'].ewm(span=7).mean()

        # Handle missing values
        self.data.fillna(method='ffill', inplace=True)  # Forward fill to handle missing values
        self.data.fillna(method='bfill', inplace=True)  # Backward fill for any remaining missing values

    def clean_data(self):
        # Convert date columns to datetime data type
        self.data['last_updated'] = pd.to_datetime(self.data['last_updated'])
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Calculate the 30-day rolling average for the 'price' column
        self.data['30_day_avg'] = self.data['price'].rolling(window=30, min_periods=1).mean()
        
        # Handle missing values for all columns
        for column in self.data.columns:
            # If the column is of numeric type, fill missing values with forward fill and then backward fill
            if pd.api.types.is_numeric_dtype(self.data[column]):
                self.data[column].fillna(method='ffill', inplace=True)
                self.data[column].fillna(method='bfill', inplace=True)
            # If the column is of object type (like strings), fill missing values with the mode (most frequent value)
            elif pd.api.types.is_object_dtype(self.data[column]):
                mode_value = self.data[column].mode()[0]
                self.data[column].fillna(value=mode_value, inplace=True)
            # If the column is of datetime type, fill missing values with the most recent date
            elif pd.api.types.is_datetime64_any_dtype(self.data[column]):
                self.data[column].fillna(method='ffill', inplace=True)
                self.data[column].fillna(method='bfill', inplace=True)

    def addDateTimeFeatures(self):
        # Additional Date Features
        self.data['hour'] = pd.to_datetime(self.data['last_updated'], unit='s').dt.hour
        self.data['minute'] = pd.to_datetime(self.data['last_updated'], unit='s').dt.minute
        self.data['second'] = pd.to_datetime(self.data['last_updated'], unit='s').dt.second

        # Cyclic encoding for hour, minute, and second
        self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        self.data['minute_sin'] = np.sin(2 * np.pi * self.data['minute'] / 60)
        self.data['minute_cos'] = np.cos(2 * np.pi * self.data['minute'] / 60)
        self.data['second_sin'] = np.sin(2 * np.pi * self.data['second'] / 60)
        self.data['second_cos'] = np.cos(2 * np.pi * self.data['second'] / 60)
        self.data['datetime'] = pd.to_datetime(self.data[['year', 'month', 'day', 'hour', 'minute', 'second']])
        self.data.set_index('datetime', inplace=True)
        
    def preprocess(self):
        self.fill_missing_values()
        self.convert_data_types()
        self.extract_date_features()
        self.compute_moving_average()
        self.clean_data()
        self.feature_engineering()
        self.addDateTimeFeatures()
        self.data.to_csv(self.output_path, index=False)
        print(f"Data preprocessed and saved to {self.output_path}!")

if __name__ == "__main__":
    INPUT_PATH = 'data/raw/bitcoin_data.csv'
    OUTPUT_PATH = 'data/processed/processed_bitcoin_data.csv'
    preprocessor = BitcoinPreprocessor(INPUT_PATH, OUTPUT_PATH)
    preprocessor.preprocess()


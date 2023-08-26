import pandas as pd
import numpy as np

def feature_engineering(df):
    # Convert 'date' column to Unix timestamp
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).astype(int) / 10**9

    # Convert 'last_updated' column to Unix timestamp
    if 'last_updated' in df.columns:
        df['last_updated'] = pd.to_datetime(df['last_updated']).astype(int) / 10**9

    # Date and Time Features
    df['day_of_week'] = pd.to_datetime(df['date'], unit='s').dt.dayofweek
    df['month'] = pd.to_datetime(df['date'], unit='s').dt.month
    df['year'] = pd.to_datetime(df['date'], unit='s').dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Lagged Features
    for lag in range(1, 8):  # for one week
        df[f'price_lag_{lag}'] = df['price'].shift(lag)

    # Rolling Window Statistics
    df['rolling_mean_7'] = df['price'].rolling(window=7).mean()
    df['rolling_std_7'] = df['price'].rolling(window=7).std()
    df['rolling_min_7'] = df['price'].rolling(window=7).min()
    df['rolling_max_7'] = df['price'].rolling(window=7).max()

    # Differences and Returns
    df['daily_return'] = df['price'].pct_change()
    df['log_return'] = np.log(df['price']).diff()

    # Exponential Weighted Features
    df['ewm_7'] = df['price'].ewm(span=7).mean()

    # Handle missing values
    df.fillna(method='ffill', inplace=True)  # Forward fill to handle missing values
    df.fillna(method='bfill', inplace=True)  # Backward fill for any remaining missing values

    # Check for any columns that still contain datetime strings
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.contains('2023-08-26').any():
            print(f"Column '{col}' still contains datetime strings!")

    return df

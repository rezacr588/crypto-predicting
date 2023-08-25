import pandas as pd

def fill_missing_values(data):
    """Fill missing values in the dataset."""
    data.fillna(method='ffill', inplace=True)
    data['tvl'].fillna(0, inplace=True)
    return data

def compute_moving_average(data, window=7):
    """Compute the moving average for the Bitcoin price."""
    data['7_day_avg'] = data['price'].rolling(window=window).mean()
    data['30_day_avg'] = data['price'].rolling(window=30).mean()

    return data


def convert_data_types(data):
    """Convert columns to appropriate data types."""
    numerical_cols = ['price', 'volume_24h', 'volume_change_24h', 'percent_change_1h', 'percent_change_24h', 
                      'percent_change_7d', 'percent_change_30d', 'percent_change_60d', 'percent_change_90d', 
                      'market_cap', 'market_cap_dominance', 'fully_diluted_market_cap', 'tvl']
    
    # Convert numerical columns to float
    for col in numerical_cols:
        data[col] = data[col].astype(float)
    
    # Convert last_updated to datetime
    data['last_updated'] = pd.to_datetime(data['last_updated'], errors='coerce')
    
    return data


def extract_date_features(data):
    """Extract date-related features from the dataset."""
    data['date'] = data['last_updated'].dt.date
    data['day'] = data['last_updated'].dt.day
    data['month'] = data['last_updated'].dt.month
    data['year'] = data['last_updated'].dt.year
    return data

def preprocess_bitcoin_data(input_path='data/raw/bitcoin_data.csv', output_path='data/processed/processed_bitcoin_data.csv'):
    # Load the raw data
    data = pd.read_csv(input_path)

    # Preprocessing steps
    data = fill_missing_values(data)
    data = convert_data_types(data)
    data = extract_date_features(data)
    data = compute_moving_average(data)  # Add this line

    # Save the processed data
    data.to_csv(output_path, index=False)
    print(f"Data preprocessed and saved to {output_path}!")

if __name__ == "__main__":
    preprocess_bitcoin_data()

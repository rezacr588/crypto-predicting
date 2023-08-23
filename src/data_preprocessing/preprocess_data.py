import pandas as pd

def preprocess_bitcoin_data(input_path='data/raw/bitcoin_data.csv', output_path='data/processed/processed_bitcoin_data.csv'):
    # Load the data
    data = pd.read_csv(input_path)

    # Handle Missing Values
    data.fillna(method='ffill', inplace=True)

    # Feature Engineering

    # Date Features
    if 'timestamp' in data.columns:
        data['date'] = pd.to_datetime(data['timestamp'])
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year

    # Technical Indicators

    # Moving Averages
    data['7_day_avg'] = data['price'].rolling(window=7).mean()
    data['30_day_avg'] = data['price'].rolling(window=30).mean()

    # Relative Strength Index (RSI)
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Price Movement Labeling
    data['price_movement'] = data['price'].diff().apply(lambda x: 'up' if x > 0 else ('down' if x < 0 else 'neutral'))

    # Drop any rows with NaN values after feature engineering
    data.dropna(inplace=True)

    # Save the processed data
    data.to_csv(output_path, index=False)
    print(f"Data preprocessed and saved to {output_path}!")

if __name__ == "__main__":
    preprocess_bitcoin_data()

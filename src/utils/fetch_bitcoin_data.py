import os
import pandas as pd
import requests

def fetch_latest_bitcoin_data(api_key, save_path='data/raw/bitcoin_data.csv'):
    """
    Fetches the latest Bitcoin data from CoinMarketCap and appends it to a CSV file.

    Parameters:
    - api_key (str): The API key for accessing CoinMarketCap data.
    - save_path (str, optional): The path to the CSV file where the data will be saved. 
                                 Defaults to 'data/raw/bitcoin_data.csv'.

    Returns:
    None
    """
    # Define the endpoint URL to get Bitcoin data
    endpoint_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }

    # Parameters to filter results for Bitcoin
    parameters = {
        'start': '1',
        'limit': '2',
        'convert': 'USD'
    }

    response = requests.get(endpoint_url, headers=headers, params=parameters)

    if response.status_code == 200:
        data = response.json()
        bitcoin_data = data['data'][0]  # Assuming Bitcoin is the first result       
        # Extract relevant fields
        btc_quote = bitcoin_data['quote']['USD']
        btc_quote['last_updated'] = bitcoin_data['last_updated']     
        # Convert the new data to a Pandas DataFrame
        new_data = pd.DataFrame([btc_quote])

        # Check if the CSV file already exists
        if os.path.exists(save_path):
            # Read the existing data
            existing_data = pd.read_csv(save_path)
            # Append the new data to the existing data
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            combined_data = new_data


        # Save the combined data to the CSV file
        combined_data.to_csv(save_path, index=False)
        print(f"Data saved successfully to {save_path}!")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

import requests
import pandas as pd

def fetch_bitcoin_data(api_key, save_path='data/raw/bitcoin_data.csv'):
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
        
        # Convert the data to a Pandas DataFrame
        df = pd.DataFrame([bitcoin_data['quote']['USD']])
        
        # Save the data to a CSV file
        df.to_csv(save_path, index=False)
        print(f"Data saved successfully to {save_path}!")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

if __name__ == "__main__":
    API_KEY = '402db71a-cbac-4ff7-8120-2053b050d2ae'
    fetch_bitcoin_data(API_KEY)

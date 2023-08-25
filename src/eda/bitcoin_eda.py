import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(processed_data_path):
    """Load the processed Bitcoin data."""
    return pd.read_csv(processed_data_path)

def plot_time_series(data):
    """Plot Bitcoin price over time."""
    plt.figure(figsize=(12, 6))
    data['date'] = pd.to_datetime(data['date'])
    plt.plot(data['date'], data['price'])
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    # plt.show()

def plot_price_distribution(data):
    """Plot the distribution of Bitcoin prices."""
    plt.figure(figsize=(8, 6))
    data['price'].hist(bins=50, edgecolor='black')
    plt.title('Distribution of Bitcoin Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.show()

def plot_correlation_heatmap(data):
    """Plot a correlation heatmap for the dataset."""

    # Exclude non-numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Calculate the correlation matrix
    correlation_matrix = numeric_data.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    # plt.show()


def plot_outliers(data):
    """Plot a box plot to identify price outliers."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(data['price'])
    plt.title('Box Plot of Bitcoin Prices')
    # plt.show()

def plot_moving_averages(data):
    """Plot short-term and long-term moving averages."""
    if '7_day_avg' not in data.columns:
        print("7_day_avg column is missing from the data. Skipping this plot.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['price'], label='Actual Price')
    plt.plot(data['date'], data['7_day_avg'], label='7-Day Moving Average', linestyle='--')
    plt.plot(data['date'], data['30_day_avg'], label='30-Day Moving Average', linestyle='--')
    plt.title('Bitcoin Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    # plt.show()

def perform_eda(processed_data_path):
    """
    Perform Exploratory Data Analysis (EDA) on the processed Bitcoin data.
    
    Parameters:
    - processed_data_path (str): Path to the processed data CSV file.
    """
    data = load_data(processed_data_path)
    if data.empty:
        print("No data available for EDA.")
        return
    
    # Basic Data Overview
    print("\nBasic Data Overview:")
    print(data.head())
    print(data.describe())
    print(data.info())
    
    plot_time_series(data)
    plot_price_distribution(data)
    plot_correlation_heatmap(data)
    plot_outliers(data)
    plot_moving_averages(data)

    # Additional plots and analyses can be added as separate functions and called here.

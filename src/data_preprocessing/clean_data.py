import pandas as pd

def clean_data(dataframe):
    # Convert date columns to datetime data type
    dataframe['last_updated'] = pd.to_datetime(dataframe['last_updated'])
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    
    # Calculate the 30-day rolling average for the 'price' column
    dataframe['30_day_avg'] = dataframe['price'].rolling(window=30, min_periods=1).mean()
    
    # Handle missing values for all columns
    for column in dataframe.columns:
        # If the column is of numeric type, fill missing values with forward fill and then backward fill
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            dataframe[column].fillna(method='ffill', inplace=True)
            dataframe[column].fillna(method='bfill', inplace=True)
        # If the column is of object type (like strings), fill missing values with the mode (most frequent value)
        elif pd.api.types.is_object_dtype(dataframe[column]):
            mode_value = dataframe[column].mode()[0]
            dataframe[column].fillna(value=mode_value, inplace=True)
        # If the column is of datetime type, fill missing values with the most recent date
        elif pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            dataframe[column].fillna(method='ffill', inplace=True)
            dataframe[column].fillna(method='bfill', inplace=True)
    
    return dataframe

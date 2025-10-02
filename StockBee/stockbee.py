# Make a pipeline of news sources.
# Filter out the noise.
# Understand its effect on the market: sentiment analysis.

# Correlation for Pairs Trading
# Look for stocks that move together.

#backtrader to backtest
# Nasdaq website for earnings calendar
# Yahoo Finance also earnings calendar

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests

# Global variables
TODAY = datetime.date.today()
START_DATE = TODAY - datetime.timedelta(days=30)  # Reduced to 30 days for testing
END_DATE = TODAY
TICKERS = ['AAPL', 'MSFT']  # Reduced to just 2 tickers for testing

# Check if a DataFrame is valid (not empty and has required columns)
def is_valid_dataframe(df, required_columns=None):
    """
    Check if a pandas DataFrame is valid for analysis.
    
    Args:
        df: pandas DataFrame to check
        required_columns: list of column names that must be present
    
    Returns:
        bool: True if DataFrame is valid, False otherwise
    """
    if df is None or df.empty:
        return False
    
    if required_columns:
        return all(col in df.columns for col in required_columns)
    
    return True

# Get the stock data for each stock without API throttling
def get_stock_data(ticker, start_date, end_date):
    try:
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        # Add timeout and retry parameters
        df = stock.history(start=start_date, end=end_date, timeout=10)
        
        if not is_valid_dataframe(df, ['Close']):
            print(f"Warning: No valid data found for {ticker}")
            return None
        
        print(f"Successfully fetched {len(df)} days of data for {ticker}")
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Calculate daily returns
def calculate_daily_returns(df):
    if not is_valid_dataframe(df, ['Close']):
        print("Warning: Invalid DataFrame provided to calculate_daily_returns")
        return None
        
    df_copy = df.copy()
    df_copy['Return'] = df_copy['Close'].pct_change()
    df_copy.dropna(inplace=True)
    return df_copy

# Calculate Mean Squared Difference of 2 stock returns
def calculate_mean_squared_difference(df1, df2):
    if not is_valid_dataframe(df1, ['Return']) or not is_valid_dataframe(df2, ['Return']):
        print("Warning: Invalid DataFrames provided to calculate_mean_squared_difference")
        return None
        
    try:
        merged = df1.join(df2, lsuffix='_1', rsuffix='_2')
        mse = ((merged['Return_1'] - merged['Return_2']) ** 2).mean()
        return mse
    except Exception as e:
        print(f"Error calculating mean squared difference: {e}")
        return None

def test_dataframe_validation():
    """
    Test function to demonstrate the is_valid_dataframe function
    """
    print("\n=== Testing DataFrame Validation Function ===")
    
    # Test with None
    print(f"None DataFrame: {is_valid_dataframe(None)}")
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    print(f"Empty DataFrame: {is_valid_dataframe(empty_df)}")
    
    # Test with DataFrame missing required columns
    df_no_close = pd.DataFrame({'Open': [1, 2, 3], 'High': [2, 3, 4]})
    print(f"DataFrame without 'Close' column: {is_valid_dataframe(df_no_close, ['Close'])}")
    
    # Test with valid DataFrame
    valid_df = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [105, 106, 107],
        'Low': [99, 100, 101],
        'Close': [104, 105, 106],
        'Volume': [1000, 1100, 1200]
    })
    print(f"Valid DataFrame: {is_valid_dataframe(valid_df, ['Close'])}")
    
    # Test calculate_daily_returns with valid data
    returns_df = calculate_daily_returns(valid_df)
    if returns_df is not None:
        print(f"Returns calculated successfully: {len(returns_df)} rows")
        print(f"Sample returns: {returns_df['Return'].head().tolist()}")
    else:
        print("Failed to calculate returns")
    
    print("=== End DataFrame Validation Test ===\n")

# Main function to run the analysis
def main():
    # Run the test function first
    test_dataframe_validation()
    
    print(f'Analyzing stock data from {START_DATE} to {END_DATE}')
    print(f'Tickers: {TICKERS}')
    print(f'Time now: {datetime.datetime.now()} ')
    stock_data = {}
    returns_data = {}

    # Fetch data for each ticker
    for ticker in TICKERS:
        df = get_stock_data(ticker, START_DATE, END_DATE)
        if df is not None:
            stock_data[ticker] = df
            returns_df = calculate_daily_returns(df)
            if returns_df is not None:
                returns_data[ticker] = returns_df
            else:
                print(f"Failed to calculate returns for {ticker}")
        else:
            print(f"Skipping {ticker} due to missing data")

    # Only proceed with pairs if we have valid data
    valid_tickers = list(returns_data.keys())
    print(f"Valid tickers with data: {valid_tickers}")
    
    if len(valid_tickers) < 2:
        print("Not enough valid tickers for pairs analysis")
        return

    # Calculate pairwise mean squared differences
    mse_results = {}
    for i in range(len(valid_tickers)):
        for j in range(i + 1, len(valid_tickers)):
            ticker1 = valid_tickers[i]
            ticker2 = valid_tickers[j]
            mse = calculate_mean_squared_difference(returns_data[ticker1], returns_data[ticker2])
            if mse is not None:
                mse_results[(ticker1, ticker2)] = mse

    # Print results
    if mse_results:
        print("\nPairwise Mean Squared Differences:")
        for pair, mse in mse_results.items():
            print(f'{pair[0]} vs {pair[1]}: {mse:.6f}')
    else:
        print("No valid pairs found for analysis")
    print(f'Analysis completed at: {datetime.datetime.now()}')
    
if __name__ == "__main__":
    main()
# add new feautres and prepare for modeling

import pandas as pd

def add_features(df):
    df = df.copy()
    
    # moving averages
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # relative strength index
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # lagged returns (1-day and 5-day)
    df['Return_1d'] = df['Close'].pct_change(periods=1)
    df['Return_5d'] = df['Close'].pct_change(periods=5)
    
    # rolling volatility (sd over 10 days)
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    # drop rows with NaN 
    df.dropna(inplace=True)
    
    return df

def prepare_model_data(df, target='Close'):
    features = ['MA10', 'MA50', 'RSI', 'Return_1d', 'Return_5d', 'Volatility']
    X = df[features]
    y = df[target]
    return X, y

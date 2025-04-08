# downloading and cleaning our data

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import sys

def download_etf_data(etf_list, start_date, end_date):
    print("Creating data directories...")
    os.makedirs("data/raw", exist_ok=True)
    data = {}
    
    for etf in etf_list:
        try:
            print(f"Downloading data for {etf} from {start_date} to {end_date}...")
            df = yf.download(etf, start=start_date, end=end_date)
            if df.empty:
                print(f"Warning: No data downloaded for {etf}")
                continue
                
            df.reset_index(inplace=True)
            data[etf] = df
            df.to_csv(f"data/raw/{etf}_raw.csv", index=False)
            print(f"Successfully downloaded {len(df)} rows for {etf}")
            
        except Exception as e:
            print(f"Error downloading {etf}: {str(e)}")
            continue
            
    if not data:
        print("Error: No data was downloaded for any ETF")
        sys.exit(1)
        
    return data

def clean_data(data_dict):
    print("Cleaning downloaded data...")
    os.makedirs("data/processed", exist_ok=True)
    cleaned_data = {}
    
    for etf, df in data_dict.items():
        try:
            print(f"Cleaning data for {etf}...")
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True)
            df.fillna(method='ffill', inplace=True)
            cleaned_data[etf] = df
            df.to_csv(f"data/processed/{etf}_processed.csv", index=False)
            print(f"Successfully cleaned data for {etf}")
            
        except Exception as e:
            print(f"Error cleaning {etf}: {str(e)}")
            continue
            
    if not cleaned_data:
        print("Error: No data was cleaned successfully")
        sys.exit(1)
        
    return cleaned_data

# downloading and cleaning our data

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

def download_etf_data(etf_list, start_date, end_date):
    os.makedirs("data/raw", exist_ok=True)
    data = {}
    for etf in etf_list:
        print(f"Downloading data for {etf} ...")
        df = yf.download(etf, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        data[etf] = df
        df.to_csv(f"data/raw/{etf}_raw.csv", index=False)
    return data

def clean_data(data_dict):
    os.makedirs("data/processed", exist_ok=True)
    cleaned_data = {}
    for etf, df in data_dict.items():
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.fillna(method='ffill', inplace=True)
        cleaned_data[etf] = df
        df.to_csv(f"data/processed/{etf}_processed.csv", index=False)
    return cleaned_data

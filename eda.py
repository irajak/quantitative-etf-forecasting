# exploratory data analysis

import os
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(cleaned_data, etf):
    df = cleaned_data[etf]
    
    os.makedirs("results", exist_ok=True)
    
    # closing prices over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'])
    plt.title(f"{etf} Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.savefig(f"results/{etf}_closing_price.png")
    plt.close()
    
    # volume over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Volume'])
    plt.title(f"{etf} Trading Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.savefig(f"results/{etf}_volume.png")
    plt.close()
    
    # correlation heatmap for numerical features
    corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title(f"{etf} feature correlation")
    plt.savefig(f"results/{etf}_correlation.png")
    plt.close()

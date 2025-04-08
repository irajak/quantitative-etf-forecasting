# backtesting

import os
import pandas as pd
import matplotlib.pyplot as plt

def backtest_strategy(df, model, feature_cols, window=30):
    df = df.copy()
    predictions = []
    actuals = []
    dates = []
    
    for i in range(window, len(df)):
        train = df.iloc[i - window:i]
        test = df.iloc[i:i + 1]
        X_train = train[feature_cols]
        y_train = train['Close']
        X_test = test[feature_cols]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions.append(pred[0])
        actuals.append(test['Close'].values[0])
        dates.append(test['Date'].values[0])
    
    results_df = pd.DataFrame({'Date': dates, 'Actual': actuals, 'Predicted': predictions})
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/backtest_results.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Date'], results_df['Actual'], label="Actual")
    plt.plot(results_df['Date'], results_df['Predicted'], label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Backtesting Results")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/backtest_plot.png")
    plt.close()
    
    return results_df

# ties everything together

from datetime import datetime
from data_collection import download_etf_data, clean_data
from eda import perform_eda
from feature_engineering import add_features, prepare_model_data
from models import train_regression_models, train_classification_models, train_arima_model, train_lstm_model
from backtesting import backtest_strategy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def main():
    # STEP 1: data parsing
    etf_list = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLC']
    start_date = "2013-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    print("Downloading and cleaning data...")
    raw_data = download_etf_data(etf_list, start_date, end_date)
    cleaned_data = clean_data(raw_data)
    
    # STEP 2: exploratory data analysis
    etf_demo = 'XLK'
    print(f"Performing EDA for {etf_demo} ...")
    perform_eda(cleaned_data, etf_demo)
    
    # STEP 3: feature engineering
    print("engineering features...")
    df_etf = cleaned_data[etf_demo]
    df_features = add_features(df_etf)
    df_features.to_csv("data/processed/XLK_features.csv", index=False)
    
    # STEP 4.1: different models
    print("Preparing data for modeling...")
    X, y = prepare_model_data(df_features, target='Close')
    
    # train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    print("Training regression models...")
    regression_results = train_regression_models(X_train, y_train, X_test, y_test)
    for model_name, metrics in regression_results.items():
        print(f"{model_name}: MSE = {metrics['MSE']:.2f}, R2 = {metrics['R2']:.2f}")
    
    print("Training classification models...")
    classification_results = train_classification_models(X_train, y_train, X_test, y_test)
    for model_name, metrics in classification_results.items():
        print(f"{model_name}: Accuracy = {metrics['Accuracy']:.2f}")
    
    # STEP 4.2 time series modeling
    print("training ARIMA model...")
    arima_model, arima_forecast = train_arima_model(df_features)
    print("ARIMA Forecast for the next 10 days:")
    print(arima_forecast)
    
    print("Training LSTM model...")
    lstm_model, scaler, lstm_predictions = train_lstm_model(df_features)
    import pandas as pd
    pd.DataFrame(lstm_predictions, columns=['LSTM_Predictions']).to_csv("results/lstm_predictions.csv", index=False)
    
    # STEP 5: backtesting
    print("backtesting strategy...")
    feature_columns = ['MA10', 'MA50', 'RSI', 'Return_1d', 'Return_5d', 'Volatility']
    backtest_df = backtest_strategy(df_features, RandomForestRegressor(n_estimators=100, random_state=42), feature_columns)
    
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()

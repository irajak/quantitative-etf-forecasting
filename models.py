# train regression, classification, ARIMA, LSTM 

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import xgboost as xgb
import statsmodels.api as sm

# LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def train_regression_models(X_train, y_train, X_test, y_test):
    """
    for the three regression models:
      - Linear Regression (baseline)
      - Random Forest Regressor (non-linear)
      - XGBoost Regressor (non-linear)
    
    :return: dictionary of model performance metrics
    """
    results = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    results['Linear Regression'] = {'model': lr, 'MSE': mse_lr, 'R2': r2_lr}
    
    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    results['Random Forest'] = {'model': rf, 'MSE': mse_rf, 'R2': r2_rf}
    
    # XGBoost Regressor
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgbr.fit(X_train, y_train)
    y_pred_xgb = xgbr.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    results['XGBoost'] = {'model': xgbr, 'MSE': mse_xgb, 'R2': r2_xgb}
    
    return results

def train_classification_models(X_train, y_train, X_test, y_test):
    """
    for the two classification models:
      - Logistic Regression (baseline)
      - Gradient Boosting Classifier (non-linear)
      
    The target is defined as:
         1 if the next day's percentage change is positive, otherwise 0.
    
    :return: dictionary of classification performance metrics
    """
    results = {}
    
    # create binary target: based on next day's return
    y_train_bin = (y_train.pct_change().shift(-1) > 0).astype(int)[:-1]
    y_test_bin = (y_test.pct_change().shift(-1) > 0).astype(int)[:-1]
    
    # align features (drop the last row to match shifted target)
    X_train_bin = X_train.iloc[:-1]
    X_test_bin = X_test.iloc[:-1]
    
    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_bin, y_train_bin)
    y_pred_lr = logreg.predict(X_test_bin)
    acc_lr = accuracy_score(y_test_bin, y_pred_lr)
    results['Logistic Regression'] = {'model': logreg, 'Accuracy': acc_lr, 
                                      'Confusion Matrix': confusion_matrix(y_test_bin, y_pred_lr)}
    
    # Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbc.fit(X_train_bin, y_train_bin)
    y_pred_gbc = gbc.predict(X_test_bin)
    acc_gbc = accuracy_score(y_test_bin, y_pred_gbc)
    results['Gradient Boosting'] = {'model': gbc, 'Accuracy': acc_gbc,
                                    'Confusion Matrix': confusion_matrix(y_test_bin, y_pred_gbc)}
    
    return results

def train_arima_model(df, order=(1, 1, 1)):
    """
    Train an ARIMA model on the closing price of the ETF.
    This function fits the model and forecasts the next 10 days.
    
    :param df: DataFrame with 'Date' and 'Close' columns
    :param order: ARIMA order (p, d, q)
    :return: Fitted ARIMA model and forecast for the next 10 days
    """
    df_arima = df.copy()
    df_arima.set_index('Date', inplace=True)
    model = sm.tsa.ARIMA(df_arima['Close'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    return model_fit, forecast

def train_lstm_model(df, feature='Close', epochs=50, batch_size=32):
    """
    Train a simple LSTM model to forecast the closing price.
    This example uses a sequence length of 10 days.
    
    :param df: DataFrame with the feature column (default 'Close')
    :param epochs: Number of training epochs
    :param batch_size: Training batch size
    :return: Trained LSTM model, the scaler used, and predictions on test set
    """
    # prepare data
    data = df[[feature]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # create sequences for LSTM input
    sequence_length = 10
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    X, y = np.array(X), np.array(y)
    
    # train/test split (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # ourLSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='loss', patience=5)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    
    # predict and inverse transform
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return model, scaler, predictions

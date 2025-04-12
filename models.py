# train regression, classification, ARIMA, LSTM 

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import xgboost as xgb
import statsmodels.tsa.arima.model as sm_arima

# LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
print(tf.__version__) 
print(tf.reduce_sum(tf.random.normal([1000, 1000])))

def train_regression_models(X_train, y_train, X_test, y_test):
    """
        X_train (pd.DataFrame or np.ndarray): Training features

        y_train (pd.Series or np.ndarray): Training target

        X_test (pd.DataFrame or np.ndarray): Test features

        y_test (pd.Series or np.ndarray): Test target

        dict: Dictionary containing trained models and performance metrics (MSE, R2)
    """
    results = {}
    
    try:
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
        
    except Exception as e:
        print(f"Error in training regression models: {str(e)}")
        return None
    
    return results

def train_classification_models(X_train, y_train, X_test, y_test):
    """
        X_train (pd.DataFrame): Training features

        y_train (pd.Series): Training target (continuous prices)

        X_test (pd.DataFrame): Test features

        y_test (pd.Series): Test target (continuous prices)
    
        dict: Dictionary containing trained models and performance metrics (Accuracy, Confusion Matrix)
    """
    results = {}
    
    try:
        # Create binary target based on next day's return
        y_train_bin = (y_train.pct_change().shift(-1) > 0).astype(int).iloc[:-1]
        y_test_bin = (y_test.pct_change().shift(-1) > 0).astype(int).iloc[:-1]
        
        # Align features with the shifted target
        X_train_bin = X_train.iloc[:-1]
        X_test_bin = X_test.iloc[:-1]
        
        # Logistic Regression
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train_bin, y_train_bin)
        y_pred_lr = logreg.predict(X_test_bin)
        acc_lr = accuracy_score(y_test_bin, y_pred_lr)
        results['Logistic Regression'] = {
            'model': logreg,
            'Accuracy': acc_lr,
            'Confusion Matrix': confusion_matrix(y_test_bin, y_pred_lr)
        }
        
        # Gradient Boosting Classifier
        gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gbc.fit(X_train_bin, y_train_bin)
        y_pred_gbc = gbc.predict(X_test_bin)
        acc_gbc = accuracy_score(y_test_bin, y_pred_gbc)
        results['Gradient Boosting'] = {
            'model': gbc,
            'Accuracy': acc_gbc,
            'Confusion Matrix': confusion_matrix(y_test_bin, y_pred_gbc)
        }
        
    except Exception as e:
        print(f"Error in training classification models: {str(e)}")
        return None
    
    return results

def train_arima_model(df, order=(1, 1, 1)):
    """
        df (pd.DataFrame): DataFrame with 'Date' and 'Close' columns

        order (tuple): ARIMA order (p, d, q), default is (1, 1, 1)
    
        tuple: Fitted ARIMA model and forecast for the next 10 days
    """
    try:
        df_arima = df.copy()
        df_arima.set_index('Date', inplace=True)
        model = sm_arima.ARIMA(df_arima['Close'], order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)
        return model_fit, forecast
    except Exception as e:
        print(f"Error in training ARIMA model: {str(e)}")
        return None, None

def train_lstm_model(df, feature='Close', sequence_length=10, epochs=50, batch_size=32):
    """

        df (pd.DataFrame): DataFrame with the feature column (default 'Close')

        feature (str): Column name to predict, default is 'Close'

        sequence_length (int): Number of past days to use as input, default is 10

        epochs (int): Number of training epochs, default is 50

        batch_size (int): Training batch size, default is 32
    
        tuple: Trained LSTM model, scaler used, and predictions on the test set
    """
    try:
        # prepare our data
        data = df[[feature]].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # create sequences for LSTM input
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i + sequence_length])
            y.append(scaled_data[i + sequence_length])
        X, y = np.array(X), np.array(y)
        
        # ensure 3D shape: (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Train/test split (80/20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # define LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        # train with early stopping
        early_stop = EarlyStopping(monitor='loss', patience=5)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
        
        # predict and inverse transform
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        return model, scaler, predictions
    
    except Exception as e:
        print(f"Error in training LSTM model: {str(e)}")
        return None, None, None
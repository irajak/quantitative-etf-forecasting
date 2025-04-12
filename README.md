Quantitative ETF Forecasting
============================

**ENSF 444: Machine Learning Systems**\
**Authors:** Iraj Akbar (30146997), Syed Waliullah (30153716)

* * * * *

Project Overview
----------------

This project focuses on developing predictive models for forecasting the performance of Exchange-Traded Funds (ETFs), particularly sector-specific ETFs linked to the S&P 500. The goal is to leverage machine learning techniques to enhance trading strategies for Quantum Capital, an algorithmic trading hedge fund.

Objectives
----------

-   Predict ETF returns using regression analysis.

-   Classify ETFs into categories (outperforming/underperforming benchmarks).

-   Utilize time series forecasting to identify trends in ETF performance.

-   Improve trading decision-making with data-driven insights.

Data
----

Historical data was retrieved from Yahoo Finance (https://ca.finance.yahoo.com/) using automated Python scripts. The dataset includes:

-   Daily price movements

-   Trading volume

-   Volatility indices

All datasets and scripts are available in this repository.

Methodology
-----------

The project employs the following machine learning models:

### Regression Models

-   **Linear Regression**: Baseline predictions for ETF returns.

-   **Random Forest Regressor**: Captures complex relationships between multiple variables.

### Classification Models

-   **Logistic Regression**: Categorizes ETFs based on relative market performance.

-   **Gradient Boosting Classifier**: Enhances accuracy by addressing classification errors.

### Time Series Models

-   **ARIMA**: Forecasts future ETF prices by identifying temporal patterns.

-   **LSTM (Long Short-Term Memory)**: Deep learning approach for complex time-dependent predictions.

Technologies Used
-----------------

-   Python

-   Pandas, NumPy

-   Scikit-learn, XGBoost, Statsmodels

-   TensorFlow, Keras

Repository Structure
--------------------

```
repo/
├── LICENSE
├── README.md
├── backtesting.py                # Script for model evaluation and backtesting
├── data_collection.py            # Script to retrieve ETF data from Yahoo Finance
├── eda.py                        # Exploratory data analysis and visualizations
├── feature_engineering.py        # Data preprocessing and feature creation
├── main.py                       # Main execution script to run the full pipeline
└── models.py                     # Contains regression, classification, and time series models

```

How to Run
----------

1.  **Setup Environment**:

```
pip install -r requirements.txt

```

1.  **Data Collection**:

```
python data_collection.py

```

1.  **Run Analysis & Models**:

```
python eda.py
python feature_engineering.py
python models.py
python backtesting.py
python main.py

```

Demo
----

For an interactive demonstration and visualization, refer to the included scripts and outputs.

Team Members
------------

-   Iraj Akbar (30146997)

-   Syed Waliullah (30153716)

References
----------

-   Yahoo Finance: <https://ca.finance.yahoo.com/>

-   Wealthsimple ETF explanation: [What is an ETF?](https://www.wealthsimple.com/en-ca/learn/what-is-etf)

-   S&P 500 Overview: [TradingView](https://www.tradingview.com/markets/indices/quotes-snp/)
  
-   Used ChatGPT AI to help with fine-tuning and finding mistakes in the code. To make improvements as well.

* * * * *

If you have any questions or contributions, don't hesitate to get in touch with the team members.





# for data reading and downloading from web

import yfinance as yf
import pandas as pd

# Fetch historical data for two stocks
stock_1 = yf.download('AAPL', start='2020-01-01', end='2024-01-01')['Adj Close']
stock_2 = yf.download('MSFT', start='2020-01-01', end='2024-01-01')['Adj Close']

# Combine into a DataFrame
data = pd.DataFrame({'AAPL': stock_1, 'MSFT': stock_2})

print(data)


# after downloading we want to use some type of cointegration test for stocks
# We can use methods such as Engle-Granger Two-step test which uses the ADF (augmented dickey fuller test)
# The Engle-Granger Two-Step method starts by creating residuals based on the static regression and then testing the residuals for the presence of unit-roots. 
# It uses the Augmented Dickey-Fuller Test (ADF) or other tests to test for stationarity units in time series.

# other types of tests for cointegration include the Johansen test

# example of a coinitegration test with statsmodels.
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller


score, p_value, _ = coint(series1, series2)
print(f'Cointegration test p-value: {p_value}')

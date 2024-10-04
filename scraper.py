



# for data reading and downloading from web

import yfinance as yf
import pandas as pd

# for data visualization
import statsmodels.api as sm

import matplotlib
# set up display to work in terminal
matplotlib.use('Agg')

import matplotlib.pyplot as plt



# Fetch historical data for two stocks
apple = yf.download('AAPL', start='2020-01-01', end='2024-01-01')['Adj Close']
microsoft = yf.download('MSFT', start='2020-01-01', end='2024-01-01')['Adj Close']
google = yf.download('GOOG', start='2020-01-01', end='2024-01-01')['Adj Close']
# Combine into a DataFrame
data = pd.DataFrame({'AAPL': apple, 'MSFT': microsoft, 'GOOG': google})

# {'AAPL': apple, 'MSFT': microsoft, 'GOOG': google}

#data.columns = ['Close']

print(data)
#print(data.columns)

# after downloading we want to use some type of cointegration test for stocks

# We can use methods such as Engle-Granger Two-step test which uses the ADF (augmented dickey fuller test)
# The Engle-Granger Two-Step method starts by creating residuals based on the static regression and then testing the residuals for the presence of unit-roots. 
# It uses the Augmented Dickey-Fuller Test (ADF) or other tests to test for stationarity units in time series.

# other types of tests for cointegration include the Johansen test

# Plot ACF
fig, ax = plt.subplots(figsize=(10, 6))


# stationarity is based on unit roots
# stationarity helps to determine if mean, variance, and autocorrelation follow a similiar pattern over time.
# relies on certain assumptions that can be proven with tests like ADF
sm.graphics.tsa.plot_acf(data['Close'], lags=40, ax=ax)
plt.title('Autocorrelation (ACF)')
plt.savefig('plot.png')

X = sm.add_constant(data['MSFT'])

model = sm.OLS(data['AAPL'], X).fit()

# Get the hedge ratio (slope)
hedge_ratio = model.params[1]

# Calculate the spread
data['Spread'] = data['AAPL'] - hedge_ratio * data['MSFT']
print(data.head())

import matplotlib.pyplot as plt

# Plot the spread
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Spread'], label='Spread')
plt.axhline(data['Spread'].mean(), color='red', linestyle='--', label='Mean')
plt.title('Spread between AAPL and MSFT')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend()
plt.show()




data['Z-score'] = (data['Spread'] - data['Spread'].mean()) / data['Spread'].std()

# Plot the Z-score
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Z-score'], label='Z-score')
plt.axhline(1, color='red', linestyle='--', label='Upper Threshold')
plt.axhline(-1, color='green', linestyle='--', label='Lower Threshold')
plt.axhline(0, color='black', linestyle='--', label='Mean')
plt.title('Z-score of the Spread')
plt.xlabel('Date')
plt.ylabel('Z-score')
plt.legend()
plt.show()


# could implement some type of 'strategy' with hypothesis testing and confidence intervals.
# Define trading signals 

data['Long Signal'] = (data['Z-score'] < -1)
data['Short Signal'] = (data['Z-score'] > 1)

print(data[['Spread', 'Z-score', 'Long Signal', 'Short Signal']].tail())
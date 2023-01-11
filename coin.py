how could you add can use the SentimentIntensityAnalyzer to analyze the sentiment of news articles or statements made by the Federal Reserve to this code ? 
import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# create SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# example statement made by the Federal Reserve
statement = "The Federal Reserve is committed to maintaining price stability and supporting the economic recovery."

# analyze the sentiment of the statement
scores = sia.polarity_scores(statement)

# print the sentiment scores
print(scores)

# get historical data for Fantom
ftm = yf.Ticker("FTM-USD")
data = ftm.history(period="max")

# calculate the short and long moving averages
short_moving_average = data['Close'].rolling(window=5).mean()
long_moving_average = data['Close'].rolling(window=20).mean()

# create a new dataframe with the moving averages
data = pd.DataFrame({'Close': data['Close'], 'short_moving_average': short_moving_average, 'long_moving_average': long_moving_average})

# determine the trading signals
data['signal'] = None
data.loc[(data['short_moving_average'] > data['long_moving_average']) & (data['short_moving_average'].shift(1) < data['long_moving_average'].shift(1)), 'signal'] = 'buy'
data.loc[(data['short_moving_average'] < data['long_moving_average']) & (data['short_moving_average'].shift(1) > data['long_moving_average'].shift(1)), 'signal'] = 'sell'

# initialize trade statistics
wins = 0
losses = 0

# iterate over the rows and record the entry/exit prices
for i, row in data.iterrows():
    if row['signal'] == 'buy':
        data.loc[i, 'entry_price'] = row['Close']
    elif row['signal'] == 'sell':
        data.loc[i, 'exit_price'] = row['Close']
        wins += 1

# calculate the profit/loss
data['profit_loss'] = (data['exit_price'] - data['entry_price']) / data['entry_price']

# print trade statistics
print(f'Wins: {wins}, Losses: {losses}')

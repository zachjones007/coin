import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# download the required resource vader_lexicon
nltk.download('vader_lexicon')

# create SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# get historical data for Fantom
ftm = yf.Ticker("FTM-USD")
data = ftm.history(period="max")

# analyze the sentiment of news articles or statements made by the Federal Reserve
# you'll likely want to use an API or web scraping to obtain the actual text of the articles or statements in your actual implementation
statements = ["The Federal Reserve is committed to maintaining price stability and supporting the economic recovery.", "The Federal Reserve is concerned about rising inflation and may raise interest rates."]
scores = [sia.polarity_scores(statement) for statement in statements]

# assign sentiment scores to the historical data DataFrame
data['sentiment'] = [scores[i % len(scores)] for i in range(len(data))]

# calculate the short and long moving averages
short_moving_average = data['Close'].rolling(window=5).mean()
long_moving_average = data['Close'].rolling(window=20).mean()

# create a new dataframe with the moving averages and sentiment scores
data = pd.DataFrame({'Close': data['Close'], 'short_moving_average': short_moving_average, 'long_moving_average': long_moving_average, 'sentiment': data['sentiment']})

# determine the trading signals
data['signal'] = None
data.loc[(data['short_moving_average'] < data['long_moving_average']) & (data['short_moving_average'].shift(1) > data['long_moving_average'].shift(1)) & (data['sentiment'].apply(lambda x: x['compound']) < 0.5), 'signal'] = 'sell'
data.loc[(data['short_moving_average'] > data['long_moving_average']) & (data['short_moving_average'].shift(1) < data['long_moving_average'].shift(1)) & (data['sentiment'].apply(lambda x: x['compound']) > 0.5), 'signal'] = 'buy'

# initialize trade statistics
wins = 0
losses = 0

# iterate over the rows and record the entry/exit prices
for i, row in data.iterrows():
    if row['signal'] == 'buy':
        data.loc[i, 'entry_price'] = row['Close']
    elif row['signal'] == 'sell':
        data.loc[i, 'exit_price'] = row['Close']
        if (data.loc[i, 'exit_price'] - data.loc[i, 'entry_price']) > 0:
            wins += 1
        else:
            losses += 1

# calculate the profit/loss
data['profit_loss'] = (data['exit_price'] - data['entry_price']) / data['entry_price']

# print trade statistics
print(f'Wins : {wins}, Losses: {losses}')

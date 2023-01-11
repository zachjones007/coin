import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# download required NLTK resources
nltk.download('vader_lexicon')

def print_sentiment(compound_score):
    if compound_score > 0:
        print("good")
    else:
        print("bad")


# create SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

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

# Extract last time buy and sell signals went off 
last_buy_signal = data[data['signal']=='buy'].index[-1]
last_sell_signal = data[data['signal']=='sell'].index[-1]

# initialize new columns to keep track of trades
data['entry_price'] = None
data['exit_price'] = None

# iterate over the rows and record the entry/exit prices
for i, row in data.iterrows():
    if row['signal'] == 'buy':
        data.loc[i, 'entry_price'] = row['Close']
    elif row['signal'] == 'sell':
        data.loc[i, 'exit_price'] = row['Close']

# calculate the profit/loss
data['profit_loss'] = data['exit_price'] - data['entry_price']

# set the initial stop-loss at a fixed percentage of the entry price
stop_loss = 0.8 # 80% of the entry price

for i, row in data.iterrows():
    if row['signal'] == 'buy':
        data.loc[i, 'entry_price'] = row['Close']
        data.loc[i, 'stop_loss'] = row['Close'] * stop_loss
    elif row['signal'] == 'sell':
        data.loc[i, 'exit_price'] = row['Close']
    elif row['Close'] > data.loc[i, 'stop_loss']:
        
# iterate over the buy trades and calculate the profit/loss
for i, row in buy_data.iterrows():
    exit_index = sell_data[sell_data.index > i].index[0]
    data.loc[exit_index, 'profit_loss'] = (data.loc[exit_index, 'exit_price'] - row['entry_price'])/row['entry_price']

# select the last 3 buy trades
last_3_buy_trades = buy_data.tail(3)

# iterate over the last 3 buy trades and print the information
for i, row in last_3_buy_trades.iterrows():
    print(f'Trade: {i}, Entry Price: {row["entry_price"]}, Exit Price: {data.loc[row.name, "exit_price"]}, Profit/Loss: {data.loc[row.name, "profit_loss"]}')

# select the last 3 sell trades
last_3_sell_trades = sell_data.tail(3)

# iterate over the last 3 sell trades and print the information
for i, row in last_3_sell_trades.iterrows():
    print(f'Trade: {i}, Entry Price: {data.loc[row.name, "entry_price"]}, Exit Price: {row["exit_price"]}, Profit/Loss: {data.loc[row.name, "profit_loss"]}')

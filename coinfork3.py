import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# Initialize the machine learning model
log_reg = LogisticRegression()

# assign sentiment scores to the historical data DataFrame
data['sentiment'] = [scores[i % len(scores)] for i in range(len(data))]

# extract compound sentiment value from dictionaries and add as new column
data['compound_sentiment'] = data['sentiment'].apply(lambda x: x['compound'])

# Add RSI column
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Create a new column for the label, 1 for an increase in price and 0 for a decrease
data['label'] = (data['Close'] > data['Close'].shift(1)).astype(int)

# Create a feature dataset, dropping unnecessary columns
features = data[['compound_sentiment', 'RSI']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features.drop(columns=['label']), features['label'], test_size=0.2)

log_reg.fit(X_train, y_train)
data['predictions'] = log_reg.predict(X_test)
data['win'] = (data['label'] == data['predictions']).astype(int)

win_percentage = data['win'].mean()
print(win_percentage)
print(data.tail())
print(X_test.head())

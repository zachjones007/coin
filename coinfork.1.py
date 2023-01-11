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

# assign sentiment scores to the historical data DataFrame
data['sentiment'] = [scores[i % len(scores)] for i in range(len(data))]

# Create a new column for the label, 1 for an increase in price and 0 for a decrease
data['label'] = (data['Close'] > data['Close'].shift(1)).astype(int)

# Create a feature dataset, dropping unnecessary columns
features = data[['sentiment', 'label']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features.drop(columns=['label']), features['label'], test_size=0.2)

# Initialize the machine learning model
log_reg = LogisticRegression()

# assign sentiment scores to the historical data DataFrame
data['sentiment'] = [scores[i % len(scores)] for i in range(len(data))]

# extract compound sentiment value from dictionaries and add as new column
data['compound_sentiment'] = data['sentiment'].apply(lambda x: x['compound'])

# Create a new column for the label, 1 for an increase in price and 0 for a decrease
data['label'] = (data['Close'] > data['Close'].shift(1)).astype(int)

# Create a feature dataset, dropping unnecessary columns
features = data[['compound_sentiment', 'label']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features.drop(columns=['label']), features['label'], test_size=0.2)

# Initialize the machine learning model
log_reg = LogisticRegression()

# Fit the model to the training data
log_reg.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = log_reg.score(X_test, y_test)
print(f'Accuracy of model : {accuracy*100}%')

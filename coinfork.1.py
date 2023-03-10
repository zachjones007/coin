# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Fetch historical data for Fantom
ftm = yf.Ticker("FTM-USD")
data = ftm.history(period="max")

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Analyze the sentiment of the statements made by the Federal Reserve
# you'll likely want to use an API or web scraping to obtain the actual text of the articles or statements in your actual implementation
statements = ["The Federal Reserve is committed to maintaining price stability and supporting the economic recovery.", "The Federal Reserve is concerned about rising inflation and may raise interest rates."]
scores = []
for statement in statements:
    input_ids = torch.tensor([tokenizer.encode(statement, add_special_tokens=True)])
    with torch.no_grad():
        output = model(input_ids)[0]
    score = torch.sigmoid(output[0,0]).item()
    scores.append(score)

# Add RSI column
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=12).mean()
avg_loss = loss.rolling(window=12).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Assign sentiment scores to the historical data DataFrame
data['sentiment'] = [scores[i % len(scores)] for i in range(len(data))]

# Create a new column for the label, 1 for an increase in price and 0 for a decrease
data['label'] = (data['Close'] > data['Close'].shift(1)).astype(int)

# Create a feature dataset, dropping unnecessary columns
features = data[['RSI','sentiment']]

# Fill missing values with mean
features = features.fillna(features.mean())
# Drop rows with missing values
features = features.dropna()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.2, random_state=42)

# Create a logistic regression model
log_reg = LogisticRegression()

# Train the model on the training data
log_reg.fit(X_train, y_train)

# Test the model's performance on the testing data
accuracy = log_reg.score(X_test, y_test)
print("Accuracy: ", accuracy)

# Predict the next day's label using the logistic regression model
last_row = data.iloc[-1]
next_day_features = [last_row['RSI'], scores[-1]]
next_day_features = pd.DataFrame([next_day_features], columns=['RSI', 'sentiment'])
next_day_features = next_day_features.fillna(features.mean())
next_day_label = log_reg.predict(next_day_features)[0]

# Print whether the market is bullish or bearish based on the predicted label
if next_day_label == 1:
    print("The market is bullish.")
else:
    print("The market is bearish.")


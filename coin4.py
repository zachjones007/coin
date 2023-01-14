#53-65
# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Fetch historical data for Fantom
ftm = yf.Ticker("FTM-USD")
data = ftm.history(period="max")

# Add RSI column
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=12).mean()
avg_loss = loss.rolling(window=12).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Create a new column for the label, 1 for an increase in price and 0 for a decrease
data['label'] = (data['Close'] > data['Close'].shift(1)).astype(int)

# Create a feature dataset, dropping unnecessary columns
features = data[['RSI']]

# fill missing values with mean
features = features.fillna(features.mean())
# drop rows with missing values
features = features.dropna()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.2)

# Initialize the logistic regression model
log_reg = LogisticRegression()

# Train the model on the training data
log_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = log_reg.predict(X_test)

# Evaluate the model's performance using accuracy score
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
# Add the last 10 trades to the dataframe
data['win_or_lose'] = 'N/A'
data['entry_price'] = 'N/A'

last_ten_rows = data.tail(10)
for i in range(len(last_ten_rows)):
    if last_ten_rows['Close'][i] > last_ten_rows['Close'][i-1]:
        data.at[i, 'win_or_lose'] = 'win'
        data.at[i, 'entry_price'] = last_ten_rows['Open'][i]
    else:
        data.at[i, 'win_or_lose'] = 'lose'
        data.at[i, 'entry_price'] = last_ten_rows['Open'][i]


# Print the last 10 rows of the dataframe
print(data.tail(10))
print(acc)
print(f"Shape of features dataframe: {features.shape}")
print(f"Missing values in features dataframe: {features.isna().sum()}")

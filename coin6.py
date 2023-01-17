# Import necessary libraries
import time
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Fetch historical data for Fantom
ftm = yf.Ticker("FTM-USD")
data = ftm.history(period="max")

# Fetch real-time data for Fantom
ftm = yf.Ticker("FTM-USD")
real_time_data = ftm.history(interval = "1m")

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

# Add the take profit function
current_price = ftm.info["regularMarketPrice"]
for i in range(len(last_ten_rows)):
    if data.at[i, 'win_or_lose'] == 'win':
        take_profit_price = data.at[i, 'entry_price'] * 1.03
        if current_price >= take_profit_price:
            print("Trade: ", data.at[i, 'win_or_lose'])
            print("Entry Price: ",data.at[i, 'entry_price'])
            print("Exit Price: ", current_price)
    if data.at[i, 'win_or_lose'] == 'lose':
        take_loss_price = data.at[i, 'entry_price'] * 0.97
        if current_price <= take_loss_price:
            print("Trade: ", data.at[i, 'win_or_lose'])
            print("Entry Price: ",data.at[i, 'entry_price'])
            print("Exit Price: ", current_price)

# Print the last 10 rows of the dataframe
print(data.tail(10))
print(acc)

print(f"Missing values in features dataframe: {features.isna().sum()}")
import time

while True:
    # Fetch real-time data
    ftm = yf.Ticker("FTM-USD")
    current_price = ftm.info["regularMarketPrice"]
    for i in range(len(last_ten_rows)):
        if data.at[i, 'win_or_lose'] == 'win':
            take_profit_price = data.at[i, 'entry_price'] * 1.03
            if current_price >= take_profit_price:
                print("Trade: ", data.at[i, 'win_or_lose'])
                print("Entry Price: ",data.at[i, 'entry_price'])
                print("Exit Price: ", current_price)
        if data.at[i, 'win_or_lose'] == 'lose':
            take_loss_price = data.at[i, 'entry_price'] * 0.97
            if current_price <= take_loss_price:
                print("Trade: ", data.at[i, 'win_or_lose'])
                print("Entry Price: ",data.at[i, 'entry_price'])
                print("Exit Price: ", current_price)
    # Wait for 3 minutes before fetching new data
    time.sleep(180)

    # Add RSI column
    delta = real_time_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=12).mean()
    avg_loss = loss.rolling(window=12).mean()
    rs = avg_gain / avg_loss
    real_time_data['RSI'] = 100 - (100 / (1 + rs))

    # Create a new column for the label, 1 for an increase in price and 0 for a decrease
    real_time_data['label'] = (real_time_data['Close'] > real_time_data['Close'].shift(1)).astype(int)

    # Create a feature dataset, dropping unnecessary columns
    features = real_time_data[['RSI']]

    # fill missing values with mean
    features = features.fillna(features.mean())
    # drop rows with missing values
    features = features.dropna()
    
    # Make predictions on the real-time data
    y_pred = log_reg.predict(features)

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

    # Add the take profit function
    current_price = ftm.info["regularMarketPrice"]
    for i in range(len(last_ten_rows)):
        if data.at[i, 'win_or_lose'] == 'win':
            take_profit_price = data.at[i, 'entry_price'] * 1.03
            if current_price >= take_profit_price:
                print("Trade1: ", data.at[i, 'win_or_lose'])
                print("Entry Price1: ",data.at[i, 'entry_price'])
                print("Exit Price1: ", current_price)
        if data.at[i, 'win_or_los1e'] == 'lose':
            take_loss_price = data.at[i, 'entry_price'] * 0.97
            if current_price <= take_loss_price:
                print("Trade1: ", data.at[i, 'win_or_lose'])
                print("Entry Price1: ",data.at[i, 'entry_price'])
                print("Exit Price1: ", current_price)

    time.sleep(180)

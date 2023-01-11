help me fix start_date = datetime.datetime(2023, 1, 1)
end_date = datetime.datetime(2024, 1, 1)
import yfinance as yf
import backtrader as bt
import ccxt
import pandas as pd

# create an instance of the exchange object
exchange = ccxt.binance()

# define the symbol and timeframe
symbol = 'BTC/USDT'
timeframe = '1d'

# download the historical ohlcv bars
ohlcv = exchange.fetch_ohlcv(symbol, timeframe)

# convert the data to a pandas DataFrame
data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# set the timestamp as the DataFrame's index
data.set_index('timestamp', inplace=True)

# Create a subclass of the Backtrader Strategy class
class MovingAverageCrossOver(bt.Strategy):
    params = (
        ('fast_period', 5),
        ('slow_period', 20),
        ('stop_loss_percent', 0.97),
        ('take_profit_percent', 1.03),
        ('stop_loss_trailing', True),
        ('take_profit_trailing', True),
        ('size', 1),
        ('commision_percent', 0.01),
        ('slippage_percent', 0.01)
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow_period)
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        position = self.position.size
        if self.fast_ma > self.slow_ma:
            if position < 0:
                self.close()
            elif position == 0:
                size = int(self.params.size * (1 - self.params.commision_percent))
                self.stop_loss = self.data.close * self.params.stop_loss_percent
                self.take_profit = self.data.close * self.params.take_profit_percent
                self.buy(size=size, exectype=bt.Order.Stop, price=self.stop_loss)

        elif self.fast_ma < self.slow_ma:
            if position > 0:
                self.close()
            elif position == 0:
                size = int(self.params.size * (1 - self.params.commision_percent))
                
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        print('%s, %s' % (dt.isoformat(), txt))

def notify_order(self, order):
    if order.status in [order.Submitted, order.Accepted]:
        return
    if order.status in [order.Completed]:
        if order.isbuy():
            self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
            (order.executed.price,
            order.executed.value,
            order.executed.comm))
        elif order.issell():
            self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
            (order.executed.price,
            order.executed.value,
            order.executed.comm))
    elif order.status in [order.Canceled, order.Margin, order.Rejected]:
        self.log('Order Canceled/Margin/Rejected')

# Create an instance of the cerebro engine
cerebro = bt.Cerebro()

# Add the strategy to cerebro
cerebro.addstrategy(MovingAverageCrossOver)

# Create a Backtrader data feed from the pandas DataFrame
data = bt.feeds.PandasData(dataname=data)

# Add the data feed to cerebro
cerebro.adddata(data)

# Set the start and end date of the strategy
start_date = datetime.datetime(2023, 1, 1)
end_date = datetime.datetime(2024, 1, 1)
cerebro.broker.set_cash(100000)

# Run the strategy
cerebro.run()

print(data)




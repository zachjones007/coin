import yfinance as yf
import pandas as pd
import backtrader as bt

btc_data = yf.download("BTC-USD", start="2015-01-01", end="2020-01-01")
print(btc_data.head())

class MovingAverageCrossOver(bt.Strategy):
    params = (
        ('fast_period', 5),
        ('slow_period', 20)
    )
    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)

    def next(self):
        position = self.position.size
        if self.fast_ma[0] > self.slow_ma[0] and position <= 0:
            self.buy()
        elif self.fast_ma[0] < self.slow_ma[0] and position >= 0:
            self.close()

#Create an instance of cerebro
cerebro = bt.Cerebro()

#Create a Backtrader data feed from the pandas DataFrame
data = bt.feeds.PandasData(dataname=btc_data)

#Add the data feed to cerebro
cerebro.adddata(data)

#Add the strategy to cerebro
cerebro.addstrategy(MovingAverageCrossOver, fast_period=5, slow_period=20)

cerebro.broker.setcash(100000.0)

#Set the commission - 0.1% ... divide by 100 to remove the %
cerebro.broker.setcommission(commission=0.001)

#Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

#Run the strategy
cerebro.run()

#Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())


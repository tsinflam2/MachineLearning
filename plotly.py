import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd

import pandas_datareader.data as web
from datetime import datetime

# df = web.DataReader("aapl", 'yahoo', datetime(2007, 10, 1), datetime(2009, 4, 1))
df = pd.read_csv('TSLA.csv', parse_dates=True, index_col=0, nrows=30)

trace = go.Candlestick(x=df.index,
                       open=df.Open,
                       high=df.High,
                       low=df.Low,
                       close=df.Close)
data = [trace]
py.iplot(data, filename='simple_candlestick')

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')
start = dt.datetime(2000,1,1)
end = dt.datetime(2017,6,23)
df = web.DataReader("NASDAQ:TSLA", 'google', start, end)

df.to_csv('TSLA.csv')
df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

df[['High', 'Low']].plot()
plt.legend(loc='best')
plt.show()
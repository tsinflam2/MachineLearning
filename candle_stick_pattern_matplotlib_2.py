import datetime as dt
import talib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import quandl
import pandas_datareader.data as web

# Disable the ignorance of print list [0 0 0 0 ... 0 0 0 0] = > [0 0 0 0 0 100 0 100 -100 0 0]
np.set_printoptions(threshold=np.nan)

def get_candle_funcs():
    funcs = {}
    for name in talib.abstract.__FUNCTION_NAMES:
        if name.startswith('CDL'):
            funcs[name] = getattr(talib, name)
    return funcs

def get_OHLC(file_name):
    global df
    global O
    global H
    global L
    global C
    df = pd.read_csv(file_name, parse_dates=True, index_col=0, nrows=30)
    O = np.array(df['Open'])
    H = np.array(df['High'])
    L = np.array(df['Low'])
    C = np.array(df['Close'])

df2 = quandl.get("WIKI/TSLA")

O2 = np.array(df2['Open'])

#
# # Candlestick Pattern Name as functions
# funcs = get_candle_funcs()
#
# print(funcs)
#
# # Get the Open, High, Low, Close price for pattern recognizing
# get_OHLC('TSLA.csv')
#
# results = {}
# for f in funcs:
#     results[f] = funcs[f](O, H, L, C)
#
# print(results['CDLHAMMER'])
#
# # integer = talib.CDL3INSIDE(O, H, L, C)
# # print(integer)
#
# # The following code is going to plot the data out
# style.use('ggplot')
#
# df_ohlc = df['Close'].resample('1D').ohlc()
# df_volume = df['Volume'].resample('1D').sum()
# # df_ohlc = df['Close'].resample('10D').ohlc()
# # df_volume = df['Volume'].resample('10D').sum()
#
# df_ohlc.reset_index(inplace=True)
# df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
#
# ax1 = plt.subplot2grid((7,1), (0,0), rowspan=5, colspan=1)
# ax2 = plt.subplot2grid((7,1), (6,0), rowspan=1, colspan=1, sharex=ax1)
# ax1.xaxis_date()
#
# candlestick_ohlc(ax1, df_ohlc.values, width=0.2, colorup='g')
# ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
# plt.show()
#
#
# # Algo Choices:
# # Logistic Regression
# # GDBT
# Random Forest
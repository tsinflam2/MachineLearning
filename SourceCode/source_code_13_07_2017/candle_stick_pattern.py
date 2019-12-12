import talib
import pandas as pd
import numpy as np
import fix_yahoo_finance as yf


def get_candle_funcs():
    funcs = {}
    for name in talib.abstract.__FUNCTION_NAMES:
        if name.startswith('CDL'):
            funcs[name] = getattr(talib, name)
    return funcs


lookback = 500 * pd.tseries.offsets.BDay()

end = pd.Timestamp.utcnow()
start = end - lookback

# data = load_bars_from_yahoo(stocks=['AAPL'],
#                             start=start, end=end)

# data = yf.download('AAPL', start, end)
# data.to_csv('AAPL.csv', parse_dates=True)
data = pd.read_csv('AAPL.csv')

O = np.array(data['Open'])
H = np.array(data['High'])
L = np.array(data['Low'])
C = np.array(data['Close'])

integer = talib.CDL3INSIDE(O, H, L, C)
print(integer)
# print(O)
# print(H)
# print(L)
# print(C)
#
# funcs = get_candle_funcs()
#
# results = {}
# for f in funcs:
#     results[f] = funcs[f](O, H, L, C)
#
# print (results)
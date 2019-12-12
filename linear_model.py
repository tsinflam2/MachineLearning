import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import pandas as pd
import datetime

# Quandl requires API key to fetch the stock data
quandl.ApiConfig.api_key = '4_oX5z6kBUsPsQgkZgXn'

def train(ticker):
    ticker_df = quandl.get("WIKI/" + ticker)
    original_closing_price = ticker_df.copy()
    # Feature preparation
    ticker_df = ticker_df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
    ticker_df['HL_PCT'] = (ticker_df['Adj. High'] - ticker_df['Adj. Low']) / ticker_df['Adj. Low'] * 100.0
    ticker_df['PCT_change'] = (ticker_df['Adj. Close'] - ticker_df['Adj. Open']) / ticker_df['Adj. Open'] * 100.0
    ticker_df = ticker_df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

    forecast_col = 'Adj. Close'
    ticker_df.fillna(value=-99999, inplace=True)

    # we want to forecast out 10% of the stock price
    forecast_out = int(math.ceil(0.1 * len(ticker_df)))

    ticker_df['label'] = ticker_df[forecast_col].shift(-forecast_out)
    # ticker_df['label'] = ticker_df[forecast_col]

    # print(ticker_df)

    # print(ticker_df)

    X = np.array(ticker_df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    ticker_df.dropna(inplace=True)

    y = np.array(ticker_df['label'])
    # y = y[:-forecast_out]

    # Training
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    # COMMENTED OUT:
    clf = svm.SVR(kernel='linear', C=100, gamma=0.1)
    # clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)

    # Confidence Score
    print('Confidence Score(Linear Regression): ' + str(confidence))

    # with open('linearregression.pickle','wb') as f:
    #     pickle.dump(clf, f)

    # pickle_in = open('linearregression.pickle','rb')
    # clf = pickle.load(pickle_in)

    forecast_set = clf.predict(X_lately)
    ticker_df['Forecast'] = np.nan

    print(forecast_set)

    last_date = ticker_df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        ticker_df.loc[next_date] = [np.nan for _ in range(len(ticker_df.columns) - 1)] + [i]

    # ticker_df['Adj. Close'].plot()
    # ticker_df['Forecast'].plot()

    print("adjust close")
    print(ticker_df['Adj. Close'])

    print("forcast")
    print(ticker_df['Forecast'])

    return original_closing_price, ticker_df, confidence


    # plt.legend(loc=4)
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.show()


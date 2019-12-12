import talib
import pandas as pd
import numpy as np
import quandl
import math
from sklearn import preprocessing, cross_validation, linear_model, neighbors


def get_candle_funcs():
    funcs = {}
    for name in talib.abstract.__FUNCTION_NAMES:
        if name.startswith('CDL'):
            funcs[name] = getattr(talib, name)
    return funcs

# Quandl requires API key to fetch the stock data
quandl.ApiConfig.api_key = '4_oX5z6kBUsPsQgkZgXn'

# Disable the ignorance of print list [0 0 0 0 ... 0 0 0 0] = > [0 0 0 0 0 100 0 100 -100 0 0]
np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_rows', None)


def train(ticker):
    data = quandl.get("WIKI/" + ticker)
    original_closing_price = data.copy()
    data['UpDown'] = data['Adj. Volume'].pct_change()
    data.fillna(0, inplace=True)
    processed_UpDown = [1 if v > 0 else -1 for v in data['UpDown']]
    # Insert a column to indicate if the stock goes up or down in that day
    data['Label'] = processed_UpDown
    data["Date"] = pd.to_datetime(data.index)

    O = np.array(data['Open'])
    H = np.array(data['High'])
    L = np.array(data['Low'])
    C = np.array(data['Close'])

    funcs = get_candle_funcs()

    results = {}
    for f in funcs:
        results[f] = funcs[f](O, H, L, C)

    candlestick_pattern_names = list(results.keys())
    results['Label'] = data['Label'].as_matrix()
    results['Date'] = data['Date'].as_matrix()


    # Create a Pandas dataframe from some data.
    # df = pd.DataFrame(data, columns=candlestick_pattern_names)

    pd.DataFrame.from_dict(results, orient='columns').to_csv('candle_pattern.csv', index=False)
    # index_col=61 means using 61th column as index (Date)
    edited_data = pd.read_csv('candle_pattern.csv', index_col=61)


    n_samples = len(edited_data)

    # we want to forecast out 10% of the stock price
    forecast_out = int(math.ceil(0.01 * n_samples))

    X = np.array(edited_data.drop(['Label'], 1))
    X = preprocessing.scale(X)

    edited_data.dropna(inplace=True)
    y = np.array(edited_data['Label'])
    X_lately = X[-forecast_out:]
    # X = X[:-forecast_out]
    # y= y[:-forecast_out]
    # print(X)
    # print(y)

    # Training
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    # logistic_classifier = linear_model.LogisticRegression(C=1e5, n_jobs=-1)
    # logistic_classifier.fit(X_train, y_train)
    # logis_confidence = logistic_classifier.score(X_test, y_test)
    # logis_forecast_set = logistic_classifier.predict(X_lately)

    # print('Confidence Score(Logistic Regression): ' + str(logis_confidence))
    # print(' logis_forecast_set (Logistic Regression): ' + str(logis_forecast_set))

    knn = neighbors.KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_confidence = knn.score(X_test, y_test)
    knn_forecast_set = knn.predict(X_lately)


    # Confidence Score
    # print('Confidence Score(Logistic Regression): ' + str(logis_confidence))
    # print('Confidence Score(knn): ' + str(knn_confidence))
    # print(forecast_set)

    return original_closing_price, knn_forecast_set, knn_confidence

# X = np.array(edited_data.drop(['Label'], 1))
# X = preprocessing.scale(X)

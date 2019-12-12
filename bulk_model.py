import talib
import pandas as pd
import numpy as np
import quandl
import math
from sklearn import preprocessing, cross_validation, linear_model, neighbors
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn import clone
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
n_estimators = 30
plot_colors = "ryb"
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration


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
    models = [DecisionTreeClassifier(max_depth=None),
              RandomForestClassifier(n_estimators=n_estimators),
              ExtraTreesClassifier(n_estimators=n_estimators),
              AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                                 n_estimators=n_estimators)]

    data = quandl.get("WIKI/" + ticker)
    # data = quandl.get("WIKI/GOOGL")

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

    # we want to forecast out 1% of the stock price
    forecast_out = int(math.ceil(0.01 * n_samples))

    X = np.array(edited_data.drop(['Label'], 1))
    X = preprocessing.scale(X)

    edited_data.dropna(inplace=True)
    y = np.array(edited_data['Label'])
    X_lately = X[-forecast_out:]
    # print(X)
    # print(y)

    # Training
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    # used_models_list = ['DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier']
    score_list = []
    forecast_set_list = []

    for model in models:
        # Train
        clf = clone(model)
        clf = model.fit(X_train, y_train)
        forecast_set = clf.predict(X_lately)

        scores = clf.score(X_train, y_train)

        score_list.append(scores)
        forecast_set_list.append(forecast_set)
        # print(scores)
        # print(forecast_set)

        # print(used_models_list)
        print(score_list)
        print(forecast_set_list)

    return original_closing_price, score_list, forecast_set_list

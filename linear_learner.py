import numpy as np

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from math import pi
import bokeh
import quandl
import linear_model

# Quandl requires API key to fetch the stock data
quandl.ApiConfig.api_key = '4_oX5z6kBUsPsQgkZgXn'

def analyse(ticker, options):
    return draw_line_chart(ticker, options)


def draw_line_chart(ticker, options):

    # GOOGL = quandl.get("WIKI/GOOGL", returns="pandas", start_date="2017-06-01", end_date="2017-07-14")
    original, trained_ticker, confidence_score = linear_model.train(ticker)

    # if user choose one of an item in options
    if options:
        # if user choose to show 10ma
        if '10ma' in options:
            original['10ma'] = original['Close'].rolling(window=10, min_periods=0).mean()
        # if user choose to show 100ma
        if '100ma' in options:
            original['100ma'] = original['Close'].rolling(window=100, min_periods=0).mean()
        # if user choose to show 250ma
        if '250ma' in options:
            original['250ma'] = original['Close'].rolling(window=250, min_periods=0).mean()

    # Reindex from date to number which start from 0 to the length of rows of stock
    trained_ticker['Date'] = trained_ticker.index
    trained_ticker.index = range(len(trained_ticker))
    original['Date'] = original.index
    original.index = range(len(original))

    # Boket plot setting
    p1 = figure(x_axis_type="datetime", title="Stock Closing Prices")
    p1.grid.grid_line_alpha=0.3
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Price'

    # Draw line
    p1.line(datetime(trained_ticker['Date']), trained_ticker['Adj. Close'], color='#A6CEE3', legend=ticker)
    p1.line(datetime(trained_ticker['Date']), trained_ticker['Forecast'], color='red', legend='Forecast')
    # p1.line(datetime(original['Date']), original['Adj. Close'], color='#33A02C', legend='Original')
    if '10ma' in original.columns:
        p1.line(datetime(original['Date']), original['10ma'], color='orange', legend='10ma')
    if '100ma' in original.columns:
        p1.line(datetime(original['Date']), original['100ma'], color='blue', legend='100ma')
    if '250ma' in original.columns:
        p1.line(datetime(original['Date']), original['250ma'], color='purple', legend='250ma')

    p1.legend.location = 'top_left'

    # googl = np.array(GOOGL['Close'])
    # googl_dates = np.array(GOOGL['Date'], dtype=np.datetime64)

    # window_size = 30
    # window = np.ones(window_size)/float(window_size)
    # googl_avg = np.convolve(googl, window, 'same')

    output_file("stocks.html", title="Linear Regression Result")

    show(gridplot([[p1]], plot_width=800, plot_height=500))  # open a browser

    # If user has chosen option of candlestick
    if 'candlestick' in options:
        inc = original.Close > original.Open
        dec = original.Open > original.Close

        w = 12 * 60 * 60 * 1000  # half day in ms

        TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

        p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, title="Candlestick")
        p.xaxis.major_label_orientation = pi / 4
        p.grid.grid_line_alpha = 0.3

        p.segment(original.Date, original.High, original.Date, original.Low, color="black")
        # Positive candle
        p.vbar(original.Date[inc], w, original.Open[inc], original.Close[inc], fill_color="#D5E1DD", line_color="black")
        # Negative candle
        p.vbar(original.Date[dec], w, original.Open[dec], original.Close[dec], fill_color="#F2583E", line_color="black")

        output_file("candlestick.html", title="candlestick.py example")

        show(p)  # open a browser

    return confidence_score, None


def datetime(x):
    return np.array(x, dtype=np.datetime64)

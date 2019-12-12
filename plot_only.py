import numpy as np

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
import bokeh
import quandl
import bulk_model

# Quandl requires API key to fetch the stock data
quandl.ApiConfig.api_key = '4_oX5z6kBUsPsQgkZgXn'


def draw_line_chart(ticker, options):
    original = quandl.get("WIKI/" + ticker)
    print(original)

    # if user choose one of an item in options
    if options:
        # if user choose to show 10ma
        if '10ma' in options:
            original['10ma'] = original['Adj. Close'].rolling(window=10, min_periods=0).mean()
        # if user choose to show 100ma
        if '100ma' in options:
            original['100ma'] = original['Adj. Close'].rolling(window=100, min_periods=0).mean()
        # if user choose to show 250ma
        if '250ma' in options:
            original['250ma'] = original['Adj. Close'].rolling(window=250, min_periods=0).mean()

    # Reindex from date to number which start from 0 to the length of rows of stock
    original['Date'] = original.index
    original.index = range(len(original))

    # Boket plot setting
    p1 = figure(x_axis_type="datetime", title="Stock Closing Prices")
    p1.grid.grid_line_alpha = 0.3
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Price'

    # Draw original line
    p1.line(datetime(original['Date']), original['Adj. Close'], color='#A6CEE3', legend=ticker)

    if '10ma' in original.columns:
        p1.line(datetime(original['Date']), original['10ma'], color='orange', legend='10ma')
    if '100ma' in original.columns:
        p1.line(datetime(original['Date']), original['100ma'], color='blue', legend='100ma')
    if '250ma' in original.columns:
        p1.line(datetime(original['Date']), original['250ma'], color='purple', legend='250ma')

    p1.legend.location = 'top_left'

    output_file("stocks.html", title="Plot Only")

    show(gridplot([[p1]], plot_width=800, plot_height=500))  # open a browser

def datetime(x):
    return np.array(x, dtype=np.datetime64)
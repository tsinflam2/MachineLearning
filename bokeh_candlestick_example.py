from math import pi

import pandas as pd

from bokeh.plotting import figure, show, output_file
import quandl

# df = pd.DataFrame(MSFT)[:50]
df = quandl.get("WIKI/GOOGL", returns="pandas", start_date="2017-06-01", end_date="2017-07-01")
# print(df.Close)
df["Date"] = pd.to_datetime(df.index)
df.index = range(len(df))
# print(df)

inc = df.Close > df.Open
dec = df.Open > df.Close


w = 12*60*60*1000 # half day in ms

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, title = "Candlestick")
p.xaxis.major_label_orientation = pi/4
p.grid.grid_line_alpha=0.3

p.segment(df.Date, df.High, df.Date, df.Low, color="black")
# Positive candle
p.vbar(df.Date[inc], w, df.Open[inc], df.Close[inc], fill_color="#D5E1DD", line_color="black")
# Negative candle
p.vbar(df.Date[dec], w, df.Open[dec], df.Close[dec], fill_color="#F2583E", line_color="black")

output_file("candlestick.html", title="candlestick.py example")

show(p)  # open a browser
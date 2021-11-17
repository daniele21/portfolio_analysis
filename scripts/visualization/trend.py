#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from typing import Text

import pandas as pd
from bokeh.plotting import figure

from bokeh.models import BooleanFilter, CDSView, HoverTool, Range1d, LinearAxis, Slider, ColumnDataSource, \
    DateRangeSlider, DataRange1d
from bokeh.palettes import Category20
from bokeh.models.formatters import NumeralTickFormatter

# Define constants
from scripts.paths import TICKER_DATA_DIR

W_PLOT = 1300
H_PLOT = 600
TOOLS = 'pan,wheel_zoom,hover,reset'

VBAR_WIDTH = 0.4
RED = Category20[7][6]
GREEN = Category20[5][4]

BLUE = Category20[3][0]
BLUE_LIGHT = Category20[3][1]

ORANGE = Category20[3][2]
PURPLE = Category20[9][8]
BROWN = Category20[11][10]


def get_ticker_df(ticker_id: Text):
    path = f'{TICKER_DATA_DIR}/{ticker_id}.csv'
    df = pd.read_csv(path, index_col=0)
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def plot_stock_price(ticker_df,
                     date_slider: DateRangeSlider, ):
    ticker_df = ticker_df
    stock = ColumnDataSource(
        data=dict(Date=[], Open=[], Close=[], High=[], Low=[], index=[]))
    stock.data = stock.from_df(ticker_df)

    p = figure(plot_width=W_PLOT, plot_height=H_PLOT, tools=TOOLS,
               title="Stock Price", toolbar_location='above')

    inc = stock.data['Close'] > stock.data['Open']
    dec = stock.data['Open'] > stock.data['Close']
    view_inc = CDSView(source=stock, filters=[BooleanFilter(inc)])
    view_dec = CDSView(source=stock, filters=[BooleanFilter(dec)])

    # map dataframe indices to date strings and use as label overrides
    p.xaxis.major_label_overrides = {
        i + int(stock.data['index'][0]): date.strftime('%b %d') for i, date in
        enumerate(pd.to_datetime(stock.data["Date"]))
    }
    p.xaxis.bounds = (stock.data['index'][0], stock.data['index'][-1])

    p.segment(x0='index', x1='index', y0='Low', y1='High', color=GREEN, source=stock, view=view_inc)
    p.segment(x0='index', x1='index', y0='Low', y1='High', color=RED, source=stock, view=view_dec)

    p.vbar(x='index', width=VBAR_WIDTH, top='Open', bottom='Close', fill_color=GREEN, line_color=BLUE,
           source=stock, view=view_inc, name="price")
    p.vbar(x='index', width=VBAR_WIDTH, top='Open', bottom='Close', fill_color=RED, line_color=RED,
           source=stock, view=view_dec, name="price")
    p.line(x='index', y='Close', color=BLUE, source=stock)

    p.legend.location = "top_left"
    p.legend.border_line_alpha = 0
    p.legend.background_fill_alpha = 0
    p.legend.click_policy = "mute"

    p.yaxis.formatter = NumeralTickFormatter(format='€ 0,0[.]00')
    p.x_range.range_padding = 0.05
    p.xaxis.ticker.desired_num_ticks = 40
    p.xaxis.major_label_orientation = 3.14 / 4

    def update_x_range(attr, x, y):
        start_date, end_date = date_slider.value
        p.x_range.start = start_date
        p.x_range.end = end_date
        p.y_range.start = min(stock.data['Close'][start_date:end_date]) - 0.5
        p.y_range.end = max(stock.data['Close'][start_date:end_date]) + 0.5

    date_slider.on_change('value', update_x_range)

    # Select specific tool for the plot
    price_hover = p.select(dict(type=HoverTool))

    # Choose, which glyphs are active by glyph name
    price_hover.names = ["price"]
    # Creating tooltips
    price_hover.tooltips = [("Datetime", "@Date{%Y-%m-%d}"),
                            ("Open", "@Open{€ 0,0.00}"),
                            ("Close", "@Close{€ 0,0.00}"),
                            ("Volume", "@Volume{(€ 0.00 a)}")]
    price_hover.formatters = {"Date": 'datetime'}

    return p


def plot_performance(stock,
                     date_slider: DateRangeSlider = None):
    p = figure(plot_width=W_PLOT, plot_height=H_PLOT, tools=TOOLS,
               title="Performance", toolbar_location='above')

    # map dataframe indices to date strings and use as label overrides
    p.xaxis.major_label_overrides = {
        i + int(stock.data['index'][0]): date.strftime('%b %d') for i, date in
        enumerate(pd.to_datetime(stock.data["Date"]))
    }
    p.xaxis.bounds = (stock.data['index'][0], stock.data['index'][-1])

    p.line(x='index', y='performance', color=BLUE, source=stock)
    p.line(x='index', y=0, color=BLUE_LIGHT, source=stock)

    p.legend.location = "top_left"
    p.legend.border_line_alpha = 0
    p.legend.background_fill_alpha = 0
    p.legend.click_policy = "mute"

    p.yaxis.formatter = NumeralTickFormatter(format='€ 0,0[.]00')
    p.x_range.range_padding = 0.05
    p.xaxis.ticker.desired_num_ticks = 40
    p.xaxis.major_label_orientation = 3.14 / 4

    if date_slider:
        def update_x_range(attr, x, y):
            start_date, end_date = date_slider.value
            p.x_range.start = start_date
            p.x_range.end = end_date
            p.y_range.start = min(stock.data['performance'][start_date:end_date]) - 0.5
            p.y_range.end = max(stock.data['performance'][start_date:end_date]) + 0.5

        date_slider.on_change('value', update_x_range)

    # Select specific tool for the plot
    price_hover = p.select(dict(type=HoverTool))

    # Choose, which glyphs are active by glyph name
    price_hover.names = ["performance"]
    # Creating tooltips
    price_hover.tooltips = [("Datetime", "@Date{%Y-%m-%d}"),
                            ("Performance", "@performance{(0.00 a)}")]
    price_hover.formatters = {"Date": 'datetime'}

    return p


def plot_ticker_volume(ticker_df,
                       date_slider: DateRangeSlider,
                       ):
    ticker_df = ticker_df
    stock = ColumnDataSource(
        data=dict(Date=[], Open=[], Close=[], High=[], Low=[], index=[]))
    stock.data = stock.from_df(ticker_df)

    w_plot, h_plot = W_PLOT, 300
    p = figure(plot_width=w_plot, plot_height=h_plot, tools=TOOLS,
               title="Stock Volume", toolbar_location='above')

    # map dataframe indices to date strings and use as label overrides
    p.xaxis.major_label_overrides = {
        i + int(stock.data['index'][0]): date.strftime('%b %d') for i, date in
        enumerate(pd.to_datetime(stock.data["Date"]))
    }
    p.xaxis.bounds = (stock.data['index'][0], stock.data['index'][-1])

    p.vbar(x='index', width=VBAR_WIDTH, top='Volume', fill_color=BLUE, line_color=BLUE,
           source=stock, name="volume")

    p.legend.location = "top_left"
    p.legend.border_line_alpha = 0
    p.legend.background_fill_alpha = 0
    p.legend.click_policy = "mute"

    p.yaxis.formatter = NumeralTickFormatter(format='€ 0,0[.]00')
    p.x_range.range_padding = 0.05
    p.xaxis.ticker.desired_num_ticks = 40
    p.xaxis.major_label_orientation = 3.14 / 4

    # Select specific tool for the plot
    price_hover = p.select(dict(type=HoverTool))

    # Choose, which glyphs are active by glyph name
    price_hover.names = ["volume"]
    # Creating tooltips
    price_hover.tooltips = [("Datetime", "@Date{%Y-%m-%d}"),
                            ("Volume", "@Volume{($ 0.00 a)}")]
    price_hover.formatters = {"Date": 'datetime'}

    def update_x_range(attr, x, y):
        start_date, end_date = date_slider.value
        p.x_range.start = start_date
        p.x_range.end = end_date
        p.y_range.start = min(stock.data['Volume'][start_date:end_date]) - 0.5
        p.y_range.end = max(stock.data['Volume'][start_date:end_date]) + 0.5

    date_slider.on_change('value', update_x_range)

    return p


def plot_stock(stock):
    p = figure(plot_width=W_PLOT, plot_height=H_PLOT, tools=TOOLS,
               title="Stock Price", toolbar_location='above')

    # Setting the second y axis range name and range
    p.extra_y_ranges = {"y_volume": Range1d(start=0, end=1000000)}

    # Adding the second axis to the plot.
    p.add_layout(LinearAxis(y_range_name="y_volume"), 'right')

    inc = stock.data['Close'] > stock.data['Open']
    dec = stock.data['Open'] > stock.data['Close']
    view_inc = CDSView(source=stock, filters=[BooleanFilter(inc)])
    view_dec = CDSView(source=stock, filters=[BooleanFilter(dec)])

    # map dataframe indices to date strings and use as label overrides
    p.xaxis.major_label_overrides = {
        i + int(stock.data['index'][0]): date.strftime('%b %d') for i, date in
        enumerate(pd.to_datetime(stock.data["Date"]))
    }
    p.xaxis.bounds = (stock.data['index'][0], stock.data['index'][-1])

    p.segment(x0='index', x1='index', y0='Low', y1='High', color=GREEN, source=stock, view=view_inc)
    p.segment(x0='index', x1='index', y0='Low', y1='High', color=RED, source=stock, view=view_dec)

    p.vbar(x='index', width=VBAR_WIDTH, top='Open', bottom='Close', fill_color=GREEN, line_color=BLUE,
           source=stock, view=view_inc, name="price")
    p.vbar(x='index', width=VBAR_WIDTH, top='Open', bottom='Close', fill_color=RED, line_color=RED,
           source=stock, view=view_dec, name="price")
    p.line(x='index', y='Close', color=BLUE, source=stock)

    # Volume
    p.vbar(x='index', width=VBAR_WIDTH, top='Volume', fill_color=BLUE, line_color=BLUE,
           source=stock, name="volume", y_range_name="y_volume")

    p.legend.location = "top_left"
    p.legend.border_line_alpha = 0
    p.legend.background_fill_alpha = 0
    p.legend.click_policy = "mute"

    p.yaxis.formatter = NumeralTickFormatter(format='€ 0,0[.]00')
    p.x_range.range_padding = 0.05
    p.xaxis.ticker.desired_num_ticks = 40
    p.xaxis.major_label_orientation = 3.14 / 4

    # Select specific tool for the plot
    price_hover = p.select(dict(type=HoverTool))

    # Choose, which glyphs are active by glyph name
    price_hover.names = ["price"]
    # Creating tooltips
    price_hover.tooltips = [("Datetime", "@Date{%Y-%m-%d}"),
                            ("Open", "@Open{€ 0,0.00}"),
                            ("Close", "@Close{€ 0,0.00}"),
                            ("Volume", "@Volume{(€ 0.00 a)}")]
    price_hover.formatters = {"Date": 'datetime'}

    return p

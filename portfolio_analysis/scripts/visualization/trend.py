#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from math import pi
from typing import Dict, Text

import pandas as pd
from bokeh.models import ColumnDataSource, BooleanFilter, CDSView, HoverTool, Range1d, LinearAxis, DateRangeSlider
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.palettes import Category20
from bokeh.plotting import figure

# Define constants
from portfolio_analysis.scripts.constants.constants import TOOLS

HEIGHT = 500
W_PLOT, H_PLOT = 1300, 600
VBAR_WIDTH = 0.7

COLORS = {
    "RED": Category20[7][6],
    "GREEN": Category20[5][4],
    "BLUE": Category20[3][0],
    "BLUE_LIGHT": Category20[3][1],
    "ORANGE": Category20[3][2],
    "PURPLE": Category20[9][8],
    "BROWN": Category20[11][10],
}


def create_plot(title: Text, x_label: Text, y_label: Text, tools=TOOLS):
    fig = figure(
        plot_width=W_PLOT, plot_height=H_PLOT, tools=tools,
        title=title, toolbar_location="above"
    )
    fig.xaxis.axis_label = x_label
    fig.yaxis.axis_label = y_label
    return fig


def create_hover_tool(tooltips: Dict, formatters: Dict = None, names: list = None):
    """Create a reusable hover tool."""
    hover = HoverTool(tooltips=list(tooltips.items()))
    if formatters:
        hover.formatters = formatters
    if names:
        hover.names = names
    return hover


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

    p.line(x='index', y='performance', color=COLORS['BLUE'], source=stock, name='performance')
    p.line(x='index', y=0, color=COLORS['BLUE_LIGHT'], source=stock)

    p.yaxis.formatter = NumeralTickFormatter(format='€ 0,0[.]00')
    p.x_range.range_padding = 0.05
    p.xaxis.ticker.desired_num_ticks = 40
    p.xaxis.major_label_orientation = 3.14 / 4

    if date_slider:
        def update_x_range(attr, x, y):
            start_date, end_date = date_slider.value
            p.x_range.start = start_date
            p.x_range.end = end_date
            p.y_range.start = min(stock.data['performance'][start_date:end_date])
            p.y_range.end = max(stock.data['performance'][start_date:end_date])

        date_slider.on_change('value', update_x_range)

    # Select specific tool for the plot
    price_hover = p.select(dict(type=HoverTool))

    # Choose, which glyphs are active by glyph name
    price_hover.names = ["performance"]
    # Creating tooltips
    price_hover.tooltips = [("Performance", "@performance{(0.00 a)}")]

    return p


def plot_stock_with_volume(stock: ColumnDataSource):
    """Create a combined stock price and volume plot with dual y-axes."""
    fig = figure(
        x_axis_type="datetime",
        title="Stock Price and Volume",
        tools=TOOLS,
        width=W_PLOT,
        height=H_PLOT,
    )

    # Define primary y-axis for stock prices
    fig.y_range = Range1d(start=min(stock.data["Low"]) * 0.9, end=max(stock.data["High"]) * 1.1)

    # Define secondary y-axis for volume
    fig.extra_y_ranges = {"volume": Range1d(start=0, end=max(stock.data["Volume"]) * 1.1)}
    fig.add_layout(LinearAxis(y_range_name="volume", axis_label="Volume"), 'right')

    # Add stock price glyphs (primary y-axis)
    inc = stock.data["Close"] > stock.data["Open"]
    dec = stock.data["Open"] > stock.data["Close"]

    # Add volume glyphs (secondary y-axis)
    fig.vbar(
        x="Date", width=VBAR_WIDTH, top="Volume",
        fill_color=COLORS["BLUE_LIGHT"], line_color=COLORS["BLUE_LIGHT"], source=stock,
        y_range_name="volume", name="volume", fill_alpha=0.5
    )

    fig.segment(
        x0="Date", x1="Date", y0="Low", y1="High",
        color=COLORS["GREEN"], source=stock, view=CDSView(source=stock, filters=[BooleanFilter(inc)]), name="price"
    )
    fig.segment(
        x0="Date", x1="Date", y0="Low", y1="High",
        color=COLORS["RED"], source=stock, view=CDSView(source=stock, filters=[BooleanFilter(dec)]), name="price"
    )
    fig.vbar(
        x="Date", top="Open", bottom="Close", width=VBAR_WIDTH,
        fill_color=COLORS["GREEN"], source=stock, view=CDSView(source=stock, filters=[BooleanFilter(inc)]), name="price"
    )
    fig.vbar(
        x="Date", top="Open", bottom="Close", width=VBAR_WIDTH,
        fill_color=COLORS["RED"], source=stock, view=CDSView(source=stock, filters=[BooleanFilter(dec)]), name="price"
    )



    # Remove default hover tools
    fig.tools = [t for t in fig.tools if not isinstance(t, HoverTool)]

    # Add hover tool for stock price
    hover_price = create_hover_tool(
        tooltips={
            "Date": "@Date{%F}",
            "Open": "@Open",
            "Close": "@Close",
            "Volume": "@Volume{(0.00 a)}",
        },
        formatters={"@Date": "datetime"},
        names=["price"]
    )
    fig.add_tools(hover_price)

    # Add hover tool for volume
    hover_volume = create_hover_tool(
        tooltips={
            "Date": "@Date{%F}",
            "Volume": "@Volume{(0.00 a)}",
        },
        formatters={"@Date": "datetime"},
        names=["volume"]
    )
    fig.add_tools(hover_volume)

    # Format axes
    fig.xaxis.axis_label = "Date"
    fig.yaxis.axis_label = "Price (€)"
    fig.yaxis.formatter = NumeralTickFormatter(format="€ 0,0.00")
    fig.xaxis.major_label_orientation = pi / 4

    return fig

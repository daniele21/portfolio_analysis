#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from typing import Text, Dict
from math import pi
from typing import Dict, Text

from bokeh.layouts import gridplot, column
from bokeh.models import Div
from bokeh.palettes import Category20c
from bokeh.transform import cumsum
import pandas as pd
from bokeh.models import ColumnDataSource, BooleanFilter, CDSView, HoverTool, Range1d, LinearAxis, DateRangeSlider
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.palettes import Category20
from bokeh.plotting import figure

# Define constants
from portfolio_analysis.scripts.constants.constants import TOOLS
from portfolio_analysis.scripts.constants.paths import TICKER_DATA_DIR

HEIGHT = 500
W_PLOT, H_PLOT = 1300, 600
VBAR_WIDTH = 0.4

COLORS = {
    "RED": Category20[7][6],
    "GREEN": Category20[5][4],
    "BLUE": Category20[3][0],
    "BLUE_LIGHT": Category20[3][1],
    "ORANGE": Category20[3][2],
    "PURPLE": Category20[9][8],
    "BROWN": Category20[11][10],
}


def get_ticker_df(ticker_id: Text):
    path = f'{TICKER_DATA_DIR}/{ticker_id}.csv'
    df = pd.read_csv(path, index_col=0)
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def create_plot(title: Text, x_label: Text, y_label: Text, tools=TOOLS):
    fig = figure(
        plot_width=W_PLOT, plot_height=H_PLOT, tools=tools,
        title=title, toolbar_location="above"
    )
    fig.xaxis.axis_label = x_label
    fig.yaxis.axis_label = y_label
    return fig


def create_hover_tool(tooltips: Dict, formatters: Dict = None):
    hover = HoverTool(tooltips=list(tooltips.items()))
    if formatters:
        hover.formatters = formatters
    return hover


def validate_and_prepare_data(df: pd.DataFrame, required_columns: list, default_values: Dict = None):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if default_values:
        for col in missing_columns:
            df[col] = default_values.get(col, 0)
    return df


def plot_stock_price(stock, x_range=None):
    from bokeh.plotting import figure
    fig = figure(
        x_axis_type="datetime",
        title="Stock Prices",
        tools="pan,box_zoom,reset,save",
        width=W_PLOT,
        height=H_PLOT,
        x_range=x_range,
    )

    # Boolean filters for increasing and decreasing days
    inc = stock.data["Close"] > stock.data["Open"]
    dec = stock.data["Open"] > stock.data["Close"]

    # Add high-low segments
    fig.segment(
        x0="Date", x1="Date", y0="Low", y1="High",
        color="green", source=stock, view=CDSView(source=stock, filters=[BooleanFilter(inc)]), name="price"
    )
    fig.segment(
        x0="Date", x1="Date", y0="Low", y1="High",
        color="red", source=stock, view=CDSView(source=stock, filters=[BooleanFilter(dec)]), name="price"
    )

    # Add open-close bars
    fig.vbar(
        x="Date", top="Open", bottom="Close", width=0.5,
        fill_color="green", source=stock, view=CDSView(source=stock, filters=[BooleanFilter(inc)]), name="price"
    )
    fig.vbar(
        x="Date", top="Open", bottom="Close", width=0.5,
        fill_color="red", source=stock, view=CDSView(source=stock, filters=[BooleanFilter(dec)]), name="price"
    )

    # Add hover tool targeting only the 'price' glyphs
    hover = HoverTool(
        tooltips=[
            ("Date", "@Date{%F}"),
            ("Open", "@Open"),
            ("Close", "@Close"),
            ("Volume", "@Volume{(0.00 a)}"),
        ],
        formatters={"@Date": "datetime"},
        names=["price"]  # Limit hover to only glyphs with name="price"
    )
    fig.add_tools(hover)

    # Format axes
    fig.xaxis.axis_label = "Date"
    fig.yaxis.axis_label = "Price"
    fig.yaxis.formatter = NumeralTickFormatter(format="€ 0,0.00")
    fig.xaxis.major_label_orientation = 3.14 / 4

    return fig


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


def plot_performances(stock_dict: Dict,
                      date_slider: DateRangeSlider = None):
    p = figure(plot_width=W_PLOT, plot_height=H_PLOT, tools=TOOLS,
               title="Performance", toolbar_location='above')

    stock_0 = stock_dict[list(stock_dict.keys())[0]]

    # map dataframe indices to date strings and use as label overrides
    p.xaxis.major_label_overrides = {
        i + int(stock_0.data['index'][0]): date.strftime('%b %d') for i, date in
        enumerate(pd.to_datetime(stock_0.data["Date"]))
    }
    p.xaxis.bounds = (stock_0.data['index'][0], stock_0.data['index'][-1])

    for ticker_id in stock_dict:
        stock = stock_dict[ticker_id]
        p.line(x='index', y='performance', source=stock, name=ticker_id)
    p.line(x='index', y=0, color=COLORS['BLUE_LIGHT'], source=stock_0)

    p.yaxis.formatter = NumeralTickFormatter(format='0,0[.]00')
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
    price_hover.names = list(stock_dict.keys())
    # Creating tooltips
    price_hover.tooltips = [("Performance", "@performance{(0.00 a)}")]

    return p


def plot_stock_with_volume(stock: ColumnDataSource):
    from bokeh.plotting import figure
    from bokeh.models import LinearAxis, Range1d, HoverTool

    # Create the figure
    fig = figure(
        x_axis_type="datetime",
        title="Stock Price and Volume",
        tools="pan,box_zoom,reset,save",
        width=W_PLOT,
        height=H_PLOT,
    )

    # Define the primary y-axis for stock prices
    fig.y_range = Range1d(start=min(stock.data["Low"]) * 0.9, end=max(stock.data["High"]) * 1.1)

    # Define the secondary y-axis for volume
    fig.extra_y_ranges = {"volume": Range1d(start=0, end=max(stock.data["Volume"]) * 1.1)}
    fig.add_layout(LinearAxis(y_range_name="volume", axis_label="Volume"), 'right')

    # Add stock price glyphs (primary y-axis)
    inc = stock.data["Close"] > stock.data["Open"]
    dec = stock.data["Open"] > stock.data["Close"]

    fig.segment(
        x0="Date", x1="Date", y0="Low", y1="High",
        color="green", source=stock, view=CDSView(source=stock, filters=[BooleanFilter(inc)]), name="price"
    )
    fig.segment(
        x0="Date", x1="Date", y0="Low", y1="High",
        color="red", source=stock, view=CDSView(source=stock, filters=[BooleanFilter(dec)]), name="price"
    )
    fig.vbar(
        x="Date", top="Open", bottom="Close", width=VBAR_WIDTH,
        fill_color="green", source=stock, view=CDSView(source=stock, filters=[BooleanFilter(inc)]), name="price"
    )
    fig.vbar(
        x="Date", top="Open", bottom="Close", width=VBAR_WIDTH,
        fill_color="red", source=stock, view=CDSView(source=stock, filters=[BooleanFilter(dec)]), name="price"
    )

    # Add volume glyphs (secondary y-axis)
    fig.vbar(
        x="Date", width=VBAR_WIDTH, top="Volume",
        fill_color=COLORS['BLUE_LIGHT'], line_color=COLORS['BLUE_LIGHT'], source=stock,
        y_range_name="volume", name="volume"
    )

    # Add hover tool for stock price
    hover_price = HoverTool(
        tooltips=[
            ("Date", "@Date{%F}"),
            ("Open", "@Open"),
            ("Close", "@Close"),
            ("Volume", "@Volume{(0.00 a)}"),
        ],
        formatters={"@Date": "datetime"},
        names=["price"]
    )
    fig.add_tools(hover_price)

    # Add hover tool for volume
    hover_volume = HoverTool(
        tooltips=[
            ("Date", "@Date{%F}"),
            ("Volume", "@Volume{(0.00 a)}"),
        ],
        formatters={"@Date": "datetime"},
        names=["volume"]
    )
    fig.add_tools(hover_volume)

    # Format axes
    fig.xaxis.axis_label = "Date"
    fig.yaxis.axis_label = "Price (€)"
    fig.yaxis.formatter = NumeralTickFormatter(format="€ 0,0.00")
    fig.xaxis.major_label_orientation = 3.14 / 4

    return fig


def plot_ticker_volume(stock, x_range=None):
    fig = figure(
        x_axis_type="datetime",
        title="Stock Volume",
        tools=TOOLS,
        width=W_PLOT,
        height=H_PLOT,
        x_range=x_range  # Synchronize x-axis range with stock price plot
    )

    # Add volume bars
    fig.vbar(
        x="Date", width=VBAR_WIDTH, top="Volume",
        fill_color=COLORS["BLUE"], line_color=COLORS["BLUE"], source=stock, name="volume"
    )

    # Add hover tool for interactivity
    hover = create_hover_tool(
        {
            "Date": "@Date{%Y-%m-%d}",
            "Volume": "@Volume{(0.00 a)}"
        },
        {"@Date": "datetime"}
    )
    fig.add_tools(hover)

    # Format axes
    fig.xaxis.axis_label = "Date"
    fig.yaxis.axis_label = "Volume"
    fig.yaxis.formatter = NumeralTickFormatter(format="0,0")
    fig.xaxis.major_label_orientation = pi / 4

    return fig


def stake_plot(stake_dict: Dict, title: Text):
    data = pd.Series(stake_dict).reset_index(name="value")
    data["angle"] = data["value"] * 2 * pi
    data["color"] = Category20c[len(stake_dict)] if len(stake_dict) > 2 else ["#3182bd", "#6baed6"][:len(stake_dict)]

    bar_fig = create_plot("Bar Chart", "Value", "Index")
    bar_fig.hbar(
        y="index", right="value", height=0.4, fill_color="color", source=data, name="value"
    )
    bar_fig.add_tools(create_hover_tool({"Stake": "@value"}))

    wedge_fig = create_plot("Pie Chart", "", "")
    wedge_fig.axis.visible = False
    wedge_fig.wedge(
        x=0, y=1, radius=0.6, start_angle=cumsum("angle", include_zero=True),
        end_angle=cumsum("angle"), line_color="white", fill_color="color",
        legend_field="index", source=data
    )
    grid = gridplot([[wedge_fig, bar_fig]])
    return column(Div(text=title, style={"font-size": "200%"}), grid)


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

    p.segment(x0='index', x1='index', y0='Low', y1='High', color=COLORS['GREEN'], source=stock, view=view_inc)
    p.segment(x0='index', x1='index', y0='Low', y1='High', color=COLORS['RED'], source=stock, view=view_dec)

    p.vbar(x='index', width=VBAR_WIDTH, top='Open', bottom='Close', fill_color=COLORS['GREEN'],
           line_color=COLORS['BLUE'],
           source=stock, view=view_inc, name="price")
    p.vbar(x='index', width=VBAR_WIDTH, top='Open', bottom='Close', fill_color=COLORS['RED'], line_color=COLORS['RED'],
           source=stock, view=view_dec, name="price")
    p.line(x='index', y='Close', color=COLORS['BLUE'], source=stock)

    # Volume
    p.vbar(x='index', width=VBAR_WIDTH, top='Volume', fill_color=COLORS['BLUE'], line_color=COLORS['BLUE'],
           source=stock, name="volume", y_range_name="y_volume")

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

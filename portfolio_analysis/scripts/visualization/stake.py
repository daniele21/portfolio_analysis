from math import pi
from typing import Dict, Text

import pandas as pd
from bokeh.layouts import gridplot, column
from bokeh.models import Div, HoverTool
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum

from portfolio_analysis.scripts.constants.constants import TOOLS

HEIGHT = 500


def stake_plot(stake_dict: Dict,
               title: Text):
    data = pd.Series(stake_dict).reset_index(name='value')
    data = data.sort_values(by='value', ascending=True)
    data['angle'] = data['value'] * 2 * pi

    # define color

    if len(stake_dict) == 1:
        data['color'] = ['#3182bd']
    elif len(stake_dict) == 2:
        data['color'] = ['#3182bd', '#6baed6']
    else:
        data['color'] = Category20c[len(stake_dict)]

    bar_fig = figure(title='Bar Chart', toolbar_location=None,
                     tools=TOOLS, tooltips="@index: @value",
                     # x_range=(-0.5, 1.0),
                     y_range=data['index'],
                     height=HEIGHT
                     )

    wedge_fig = figure(title='Pie Chart', toolbar_location=None,
                       tools="hover", tooltips="@index: @value",
                       # x_range=(-0.5, 1.0),
                       height=HEIGHT
                       )

    wedge_fig.axis.axis_label = None
    wedge_fig.axis.visible = False
    wedge_fig.grid.grid_line_color = None

    wedge_fig.wedge(x=0, y=1, radius=0.6,
                    start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                    line_color="white", fill_color='color', legend_field='index', source=data)
    bar_fig.hbar(y='index', right='value',
                 left=0, height=0.4,
                 fill_color="color",
                 source=data,
                 name='value')

    # Select specific tool for the plot
    hover = bar_fig.select(dict(type=HoverTool))

    # Choose, which glyphs are active by glyph name
    hover.names = ["value"]
    # Creating tooltips
    hover.tooltips = [("Stake", "@value{( 0.00 )}")]

    #grid = gridplot([[wedge_fig, bar_fig]])
    grid = wedge_fig
    title = Div(text=title, style={'font-size': '200%'}, align='center')
    fig = column(title, grid)

    return fig

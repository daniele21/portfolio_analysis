from typing import Text

import numpy as np
import pandas as pd
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category20, YlGn
from bokeh.plotting import figure

from portfolio_analysis.core.operations.optimizations import msr, gmv
from portfolio_analysis.core.operations.time_series import portfolio_return, portfolio_vol
from portfolio_analysis.scripts.constants.constants import TOOLS

H_PLOT = 500
W_PLOT = 1000

VBAR_WIDTH = 0.4
RED = Category20[7][6]
GREEN = Category20[5][4]

BLUE = Category20[3][0]
BLUE_LIGHT = Category20[3][1]

ORANGE = Category20[3][2]
PURPLE = Category20[9][8]
BROWN = Category20[11][10]

GOLD = YlGn[3]


def optimization_plot(ef: pd.DataFrame,
                      er, cov,
                      title: Text,
                      riskfree_rate: float = 0.03,
                      ):
    ef['volatility'] = ef.index
    source = ColumnDataSource({col: ef[col] for col in ef.columns if col != 'weights'})

    p = figure(plot_width=W_PLOT, plot_height=H_PLOT, tools=TOOLS,
               title=title, toolbar_location='above')
    p.line(y='returns', x='volatility', color=BLUE, name='ef', source=source)

    w_msr = msr(riskfree_rate, er, cov)
    r_msr = portfolio_return(w_msr, er)
    vol_msr = portfolio_vol(w_msr, cov)
    # add CML
    cml_x = [0, vol_msr]
    cml_y = [riskfree_rate, r_msr]
    p.scatter(x=cml_x[0], y=cml_y[0], color=BROWN,
              legend_label='Risk Free', size=10)
    p.scatter(x=cml_x[1], y=cml_y[1], color=GREEN,
              legend_label='Max Sharpe Ratio', size=10)

    n = er.shape[0]
    w_ew = np.repeat(1 / n, n)
    r_ew = portfolio_return(w_ew, er)
    vol_ew = portfolio_vol(w_ew, cov)
    # add EW
    p.scatter(x=vol_ew, y=r_ew, color='#F0E442',
              legend_label='Equal Weights Portfolio', size=10)

    w_gmv = gmv(cov)
    r_gmv = portfolio_return(w_gmv, er)
    vol_gmv = portfolio_vol(w_gmv, cov)
    # add EW
    p.scatter(x=vol_gmv, y=r_gmv, color=BLUE,
              legend_label='Min Volatility', size=10)

    # Select specific tool for the plot
    ef_hover = p.select(dict(type=HoverTool))

    # Choose, which glyphs are active by glyph name
    ef_hover.names = ["ef"]
    # Creating tooltips
    ef_hover.tooltips = [("Return", "@returns{0.000}"),
                         ("Volatility", "@volatility{0.000}")]
    for col in ef.columns:
        if col not in ['returns', 'volatility', 'weights']:
            ef_hover.tooltips.append((col, f"@{{{col}}}{{0.000}}"))

    p.legend.location = "top_left"

    return p

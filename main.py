from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, DateRangeSlider

from scripts.paths import TICKER_DETAILS_PATH, TICKER_DATA_DIR, TRANSACTION_PATH
from scripts.portfolio.portfolio import Portfolio
from scripts.portfolio.tickers import Tickers
from scripts.visualization.dashboard import FinanceDashboard
from scripts.visualization.panel import tab_figures

tickers = Tickers(TICKER_DETAILS_PATH, TICKER_DATA_DIR)
portfolio = Portfolio(TRANSACTION_PATH)
dashboard = FinanceDashboard(tickers, portfolio)

fig = tab_figures({
    'Stake': dashboard.stake_status_plot(),
    'Performance': dashboard.ticker_performance_plot(),
    "History": dashboard.ticker_data_plot()
})

# curdoc().theme = 'dark_minimal'
curdoc().add_root(fig)
curdoc().title = "Portfolio Analysis"

import argparse

from bokeh.io import curdoc, output_file, save

from core.portfolio.portfolio import Portfolio
from core.portfolio.tickers import Tickers
from scripts.constants.paths import TICKER_DATA_DIR
from scripts.visualization.dashboard import FinanceDashboard
from scripts.visualization.panel import tab_figures


def main(arguments):
    title = 'Portfolio Analysis'

    ticker_details_path = arguments.tickers
    transactions_path = arguments.transactions

    tickers = Tickers(ticker_details_path, TICKER_DATA_DIR)
    portfolio = Portfolio(transactions_path)
    dashboard = FinanceDashboard(tickers, portfolio)
    tickers.update_tickers_data()

    fig = tab_figures({
        'Stake': dashboard.stake_status_plot(),
        "History": dashboard.ticker_data_plot(),
        'Performance': dashboard.ticker_performance_plot(),
        "Optimization": dashboard.portfolio_optimization()
    })

    # Generate output html
    if arguments.output is not None:
        output_file(filename=arguments.output, title=title)
        save(fig)
        print(f' > Output saved at {arguments.output}')

    # Running Server
    else:
        # curdoc().theme = 'dark_minimal'
        curdoc().add_root(fig)
        curdoc().title = title


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--tickers', help='Tickers detail filepath')
    parser.add_argument('-t', '--transactions', help='Transactions csv filepath')
    parser.add_argument('-o', '--output', help='Output file')

    args = parser.parse_args()

    main(args)

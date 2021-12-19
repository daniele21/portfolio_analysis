import argparse

from bokeh.io import output_file, save

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

    if arguments.update:
        tickers.update_tickers_data()

    fig = tab_figures({
        'Stake': dashboard.stake_status_plot(),
        "History": dashboard.ticker_data_plot(),
        'Performance': dashboard.ticker_performance_plot(),
        "Optimization": dashboard.portfolio_optimization()
    })

    # Generate output html
    output_file(filename=arguments.output, title=title)
    save(fig)
    print(f' > Output saved at {arguments.output}')

    # Running Server
    # else:
    #     # curdoc().theme = 'dark_minimal'
    #     curdoc().add_root(fig)
    #     curdoc().title = title


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--tickers', required=True, help='Tickers detail filepath')
    parser.add_argument('-t', '--transactions', required=True,  help='Transactions csv filepath')
    parser.add_argument('-o', '--output', required=True, default='outputs/portfolio_analysis.html',  help='Output filepath')
    parser.add_argument('-u', '--update', required=True, action='store_true', help='Update historical data')

    args = parser.parse_args()

    main(args)

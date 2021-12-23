import argparse

from bokeh.io import output_file, save

from portfolio_analysis.core.portfolio.portfolio import Portfolio
from portfolio_analysis.core.portfolio.tickers import Tickers
from portfolio_analysis.scripts.constants.paths import TICKER_DATA_DIR, DATA_DIR
from portfolio_analysis.scripts.utils.logging import setup_logger
from portfolio_analysis.scripts.utils.os_manager import ensure_folder
from portfolio_analysis.scripts.visualization.dashboard import FinanceDashboard
from portfolio_analysis.scripts.visualization.panel import tab_figures

logger = setup_logger('Main App')


def main(tickers_path, transactions_path, output, update=None):
    title = 'Portfolio Analysis'

    ticker_details_path = tickers_path
    transactions_path = transactions_path
    ensure_folder(DATA_DIR)
    ensure_folder(TICKER_DATA_DIR)

    tickers = Tickers(ticker_details_path, TICKER_DATA_DIR)
    portfolio = Portfolio(transactions_path)
    dashboard = FinanceDashboard(tickers, portfolio)

    if update:
        tickers.update_tickers_data()

    fig = tab_figures({
        'Stake': dashboard.stake_status_plot(),
        "History": dashboard.ticker_data_plot(),
        'Performance': dashboard.ticker_performance_plot(),
        "Optimization": dashboard.portfolio_optimization()
    })

    # Generate output html
    if output is not None:
        output_file(filename=output, title=title)
        save(fig)
        logger.info(f' > Output saved at {output}')

    # Running Server
    # else:
    #     # curdoc().theme = 'dark_minimal'
    #     curdoc().add_root(fig)
    #     curdoc().title = title

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--tickers', required=True, help='Tickers detail filepath')
    parser.add_argument('-t', '--transactions', required=True, help='Transactions csv filepath')
    parser.add_argument('-o', '--output', required=False, default='outputs/portfolio_analysis.html',
                        help='Output filepath')
    parser.add_argument('-u', '--update', required=False, action='store_true', help='Update historical data')

    args = parser.parse_args()

    main(tickers_path=args.tickers,
         transactions_path=args.transactions,
         output=args.output,
         update=args.update)

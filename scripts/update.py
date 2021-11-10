import logging
import pandas as pd
from scripts.data.ticker import add_ticker_data
from scripts.data.utils import get_ticker_list
from scripts.paths import TICKER_DATA_DIR, TRANSACTION_PATH, TICKER_DETAILS_PATH

logger = logging.getLogger('Update data')


def update_tickers_data():
    ticker_list = get_ticker_list()

    for ticker_id in ticker_list:
        logger.info(f' > Ticker: {ticker_id}')
        ticker_df = add_ticker_data(ticker=ticker_id)


def update_excel_output():
    ticker_list = get_ticker_list()
    finance_excel = 'resources/finance.xlsx'
    writer = pd.ExcelWriter(finance_excel, engine='xlsxwriter')

    # Add ticker details data
    ticker_details_df = pd.read_csv(TICKER_DETAILS_PATH, index_col=0)
    ticker_details_df.to_excel(writer, sheet_name='tickers')

    # Add transaction data
    transaction_df = pd.read_csv(TRANSACTION_PATH, index_col=0)
    transaction_df.to_excel(writer, sheet_name='transactions')

    # Grouping ticker data
    for ticker in ticker_list:
        ticker_data_path = f'{TICKER_DATA_DIR}/{ticker}.csv'
        ticker_df = pd.read_csv(ticker_data_path, index_col=0)
        ticker_df.to_excel(writer, sheet_name=ticker)

    writer.save()


if __name__ == '__main__':
    # update_tickers_data()
    update_excel_output()

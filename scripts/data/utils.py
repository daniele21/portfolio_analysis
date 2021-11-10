import os
from typing import Text
import pandas as pd

from scripts.config import BASE_DATE
from scripts.paths import TICKER_DATA_DIR, TICKER_DETAILS_PATH


def get_latest_date(ticker: Text):
    ticker_data_path = f'{TICKER_DATA_DIR}/{ticker}.csv'

    if os.path.exists(ticker_data_path):
        ticker_df = pd.read_csv(ticker_data_path, parse_dates=True)
        last_date = ticker_df.index.to_list()[-1]

    else:
        last_date = BASE_DATE

    return last_date


def get_ticker_list():
    ticker_details_df = pd.read_csv(TICKER_DETAILS_PATH, index_col=0)
    ticker_list = ticker_details_df['ticker_id'].to_list()

    return ticker_list

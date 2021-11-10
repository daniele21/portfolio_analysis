import logging
from typing import Text
import pandas as pd

from scripts.paths import TICKER_DETAILS_PATH

logger = logging.getLogger('Check')


def check_ticker_existence(ticker_id: Text):
    ticker_details_path = TICKER_DETAILS_PATH

    tickers_details_df = pd.read_csv(ticker_details_path)
    if ticker_id not in tickers_details_df['ticker_id'].to_list():
        error = f' > No valid ticker id. Register ticker in the ticker_details.csv!'
        logger.error(error)
        raise Exception(error)



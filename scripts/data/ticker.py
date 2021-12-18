import logging
import os
from typing import Text

import pandas as pd

from scripts.constants.paths import TICKER_DATA_DIR, TICKER_DETAILS_PATH
from scripts.data.check import check_ticker_existence
from scripts.data.yahoo_extraction import extract_data
from scripts.utils.date import today

logger = logging.getLogger('Update Data')


def add_ticker_data(ticker_id: Text):
    ticker_data_path = f'{TICKER_DATA_DIR}/{ticker_id}.csv'
    check_ticker_existence(ticker_id=ticker_id)

    if os.path.exists(ticker_data_path):
        ticker_df = pd.read_csv(ticker_data_path,
                                index_col=0,
                                parse_dates=True)
        last_date = str((ticker_df.index.to_list()[-1]).date())
    else:
        ticker_df = pd.DataFrame()
        last_date = None

    if str(today()) == last_date:
        return ticker_df

    # ticker_update_df = extract_data_from_yahoo(ticker=ticker_id,
    #                                            start_date=last_date)
    ticker_update_df = extract_data(ticker=ticker_id,
                                    start_date=last_date)

    ticker_df = ticker_df.append(ticker_update_df)
    ticker_df.to_csv(ticker_data_path)
    logger.info(f' > Ticker {ticker_id} data stored at {ticker_data_path}')

    return ticker_df


def add_ticker_details(ticker_id: Text,
                       ticker_name: Text,
                       isin: Text,
                       instrument: Text,
                       risk: float,
                       fee: float,
                       ticker_details_path: TICKER_DETAILS_PATH
                       ):
    if os.path.exists(ticker_details_path):
        ticker_details_df = pd.read_csv(ticker_details_path,
                                        index_col=0)
        ticker_df = ticker_details_df[ticker_details_df['ticker_id'] == ticker_id]
    else:
        ticker_details_df = pd.DataFrame()
        ticker_df = ticker_details_df

    if len(ticker_df) > 0:
        ticker_details_df = ticker_details_df.drop(ticker_details_df[ticker_details_df['ticker_id'] == ticker_id].index)

    ticker_details_df = ticker_details_df.append(pd.DataFrame({'ticker_id': ticker_id,
                                                               'ticker_name': ticker_name,
                                                               'isin': isin,
                                                               'instrument': instrument,
                                                               'risk': risk,
                                                               'fee': fee}, index=[0]))
    ticker_details_df = ticker_details_df.reset_index(drop=True)

    ticker_details_df.to_csv(ticker_details_path)
    logger.info(f' > Ticker {ticker_id} details stored at {ticker_details_path}')

    return ticker_details_df

import datetime as dt
import logging
from datetime import datetime
from typing import List, Text

import pandas as pd
import pandas_datareader as pdr
from tqdm import tqdm
from yfinance import Ticker

from scripts.config import BASE_DATE
from scripts.paths import DATA_DIR
from scripts.utils.date import readable_timestamp, today

# HISTORICAL_DATA_DF = load_data(DATA_DIR)

logger = logging.getLogger('Data Extraction')


def extract_data_from_yahoo(ticker: Text,
                            start_date: Text):
    """

    :param ticker:      Yahoo Ticker
    :param start_date:  yyyy-mm-dd
    :return:
    """

    str_today = str(today())
    start_date = start_date if start_date else BASE_DATE
    if str_today == start_date:
        return None

    ticker = Ticker(ticker=ticker)
    ticker_data = ticker.history(start=start_date)
    ticker_data.index = pd.to_datetime(ticker_data.index, format='yyyy-mm-dd')
    logger.info(f' > Loading {ticker} data from Yahoo Finance')

    return ticker_data


def extract_data(ticker: Text,
                 start_date: Text = None):
    # start_date = start_date if start_date else BASE_DATE

    data = pdr.get_data_yahoo(ticker, start=start_date)
    return data


def extract_ticker(ticker: Text,
                   period: Text = '3y'):
    if HISTORICAL_DATA_DF is not None:
        tickers = list(HISTORICAL_DATA_DF.columns)

        today = datetime.today().date()
        last_date = pd.to_datetime(max(HISTORICAL_DATA_DF[ticker].index)).date() if ticker in tickers else None

        if not last_date:
            ticker_data = Ticker(ticker)
            data_df = ticker_data.history(period=period)['Close']
            data_df.index = pd.to_datetime(data_df.index, format='yyyy-mm-dd')
            logger.info(f' > Loading {ticker} data from Yahoo Finance')

        elif last_date != today:
            # Integrate the missing datetime
            data_df = Ticker(ticker).history(start=last_date + dt.timedelta(1))
            data_df.index = pd.to_datetime(data_df.index, format='yyyy-mm-dd')
            data_df = HISTORICAL_DATA_DF[ticker].append(data_df['Close'])
            logger.info(f' > Update data for ticker {ticker} up to today')

        elif last_date == today:
            data_df = HISTORICAL_DATA_DF[ticker]
            logger.info(f' > Loading {ticker} data from Cache')

        else:
            raise AttributeError(' > No valid option')

    else:
        ticker_data = Ticker(ticker)
        data_df = ticker_data.history(period=period)['Close']
        data_df.index = pd.to_datetime(data_df.index, format='yyyy-mm-dd')
        logger.info(f' > Loading {ticker} data from Yahoo Finance')

    return data_df


def extract_tickers(ticker_list: List,
                    period: Text = '3y',
                    save_dir=DATA_DIR):
    history_df = pd.DataFrame()

    for ticker in tqdm(ticker_list, desc=' > Loading tickers: '):
        ticker_df = extract_ticker(ticker, period)
        ticker_df = ticker_df.to_frame(ticker)

        history_df = history_df.merge(ticker_df[ticker].to_frame(ticker),
                                      how='outer',
                                      left_index=True,
                                      right_index=True)

    if save_dir:
        filename = f'data_{readable_timestamp()}.csv'
        filepath = f'{save_dir}/{filename}'
        history_df.to_csv(filepath)
        logger.info(f' > Ticker Data saved at {filepath}')

    return history_df

import logging
import os
from typing import Text

import pandas
import pandas as pd

from scripts.constants.paths import TICKER_DETAILS_PATH, TRANSACTION_PATH

logger = logging.getLogger('Load Data')


def load_csv(df_path: Text):
    if os.path.exists(df_path):
        return pd.read_csv(df_path, index_col=0, parse_dates=True)
    else:
        error = f' > No file {df_path} found!'
        logger.error(f' > No file {df_path} found!')
        raise FileNotFoundError(error)


def load_tickers_details():
    ticker_details_path = TICKER_DETAILS_PATH
    return load_csv(ticker_details_path)


def load_ticker_data(ticker_id: Text,
                     ticker_data_folder: Text):
    ticker_path = f'{ticker_data_folder}/{ticker_id}.csv'
    return load_csv(ticker_path)


def load_ticker_detail(ticker_id: Text):
    ticker_details_df = load_tickers_details()
    return get_ticker_detail(ticker_id, ticker_details_df)


def get_ticker_detail(ticker_id: Text,
                      ticker_details_df: pd.DataFrame):
    try:
        ticker_detail = ticker_details_df[ticker_details_df['ticker_id'] == ticker_id]
        return ticker_detail.to_dict()
    except Exception as e:
        logger.error(f' > No valid ticker id: {ticker_id} - {e}')
        raise Exception(e)


def load_transactions():
    transaction_path = TRANSACTION_PATH
    return load_csv(transaction_path)


def load_ticker_transactions(ticker_id: Text):
    transaction_df = load_transactions()
    return get_ticker_transactions(ticker_id, transaction_df)


def get_ticker_transactions(ticker_id: Text,
                            transactions_df: pd.DataFrame):
    try:
        ticker_tran_df = transactions_df[transactions_df['ticker_id'] == ticker_id]
        return ticker_tran_df
    except Exception as e:
        logger.error(f' > No valid ticker_id {ticker_id} in transactions - {e}')
        raise Exception(e)


# ----------------------------------------------------------------------------------------


def get_recent_file(folder: Text,
                    contains: Text = None):
    filename_list = []

    if contains:
        for filename in os.listdir(folder):
            if contains in filename:
                filename_list.append(filename)
    else:
        for filename in os.listdir(folder):
            filename_list.append(filename)

    if len(filename_list) > 0:
        return sorted(filename_list)[-1]
    else:
        return None


def get_latest_data_path(folder: Text):
    recent_file = get_recent_file(folder, contains='data')
    file_path = f'{folder}/{recent_file}' if recent_file else None

    return file_path


def get_latest_port_performance_path(folder: Text):
    recent_file = get_recent_file(folder, contains='portfolio_performance')
    file_path = f'{folder}/{recent_file}' if recent_file else None

    return file_path


def get_latest_ticker_performance_path(folder: Text):
    recent_file = get_recent_file(folder, contains='ticker_performance')
    file_path = f'{folder}/{recent_file}' if recent_file else None

    return file_path


def load_data(folder: Text):
    data_path = get_latest_data_path(folder)

    if data_path:
        return pandas.read_csv(data_path,
                               index_col=0,
                               parse_dates=True)
    else:
        return None


def load_portfolio_performance(folder: Text):
    perf_path = get_latest_port_performance_path(folder)

    if perf_path:
        return pandas.read_csv(perf_path,
                               index_col=0,
                               parse_dates=True)
    else:
        return None


def load_ticker_performance(folder: Text):
    perf_path = get_latest_ticker_performance_path(folder)

    if perf_path:
        return pandas.read_csv(perf_path,
                               index_col=0,
                               parse_dates=True)
    else:
        return None

import json
import logging
import os
from typing import Text

import pandas as pd

logger = logging.getLogger('Load Data')


def load_json(file_path: Text):
    if not os.path.exists(file_path):
        error = f' > No file {file_path} found!'
        logger.error(error)
        raise FileNotFoundError(error)

    with open(file_path, 'r') as f:
        json_dict = json.load(f)
        f.close()

    return json_dict


def load_csv(df_path: Text):
    if os.path.exists(df_path):
        return pd.read_csv(df_path, index_col=0, parse_dates=True)
    else:
        error = f' > No file {df_path} found!'
        logger.error(f' > No file {df_path} found!')
        raise FileNotFoundError(error)


def load_ticker_details(file_path: Text):
    ticker_details = load_json(file_path)

    details_dict = {'ticker_id': [],
                    'ticker_name': [],
                    'isin': [],
                    'instrument': [],
                    'risk': [],
                    'fee': []}

    for x in ticker_details:
        for key in x:
            details_dict[key].append(x[key])

    return pd.DataFrame(details_dict)


def load_portfolio_transactions(file_path: Text):
    transactions = load_json(file_path)

    transactions_dict = {'date': [],
                         'ticker_id': [],
                         'action': [],
                         'quantity': [],
                         'price': [],
                         'commission': [],
                         'gain': [],
                         'deposit': []}

    for x in transactions:
        for key in x:
            transactions_dict[key].append(x[key])

    return pd.DataFrame(transactions_dict)

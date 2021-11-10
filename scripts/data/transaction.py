import logging
import os
import re
from typing import Text
import pandas as pd

from scripts.data.check import check_ticker_existence
from scripts.paths import TRANSACTION_PATH, TICKER_DATA_DIR, TICKER_DETAILS_PATH

logger = logging.getLogger('Transaction')


def add_transaction(date: Text,
                    ticker_id: Text,
                    action: Text,
                    quantity: int,
                    price: float,
                    commission: float,
                    gain: float,
                    deposit: float,
                    transaction_path: Text = TRANSACTION_PATH):

    date_match = re.match(r'^(\d{4})-(\d{2})-(\d{2})', date)
    if not date_match:
        error = f' > No valid date format. Give YYYY-MM-DD'
        logger.error(error)
        raise Exception(error)

    action = action.lower()
    if action not in ['buy', 'sell', 'deposit']:
        error = f' > No valid action: {action}'
        logger.error(error)
        raise Exception(error)

    if not os.path.exists(transaction_path):
        transaction_df = pd.DataFrame()
    else:
        transaction_df = pd.read_csv(transaction_path, index_col=0)

    if action != 'deposit':
        check_ticker_existence(ticker_id=ticker_id)

    row = {'ticker_id': ticker_id,
           'action': action,
           'quantity': quantity,
           'price': price,
           'commission': commission,
           'gain': gain,
           'deposit': deposit}

    transaction_df = transaction_df.append(pd.DataFrame(row, index=[0])) \
                                   .reset_index(drop=True)

    transaction_df.to_csv(transaction_path)
    logger.info(f' > Ticker {ticker_id} transaction inserted at {transaction_path}')

    return transaction_df

import logging
from typing import Text

import pandas_datareader as pdr

logger = logging.getLogger('Data Extraction')


def extract_data(ticker: Text,
                 start_date: Text = None):
    # start_date = start_date if start_date else BASE_DATE

    data = pdr.get_data_yahoo(ticker, start=start_date)
    return data

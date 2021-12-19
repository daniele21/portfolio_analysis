import logging
from typing import Text

import pandas_datareader as pdr

logger = logging.getLogger('Data Extraction')


def extract_data(ticker: Text,
                 start_date: Text = None):

    data = pdr.get_data_yahoo(ticker, start=start_date)
    return data

import logging
from typing import Text

import pandas_datareader as pdr

from scripts.utils.date import yesterday

logger = logging.getLogger('Data Extraction')


def extract_data(ticker: Text,
                 start_date: Text = None,
                 end_date: Text = None):

    end_date = yesterday() if end_date is None else end_date

    data = pdr.get_data_yahoo(ticker,
                              start=start_date,
                              end=end_date)
    return data

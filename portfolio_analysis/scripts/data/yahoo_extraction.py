import datetime
import logging
from typing import Text

import pandas_datareader as pdr

from portfolio_analysis.scripts.utils.date import yesterday

logger = logging.getLogger('Data Extraction')


def extract_data(ticker: Text,
                 start_date: Text = None,
                 end_date: Text = None):
    date_format = '%Y-%m-%d'

    if end_date is None:
        end_date = str(yesterday())
        end_datetime = datetime.datetime.strptime(end_date, date_format)

        if start_date is not None:
            start_datetime = datetime.datetime.strptime(start_date, date_format)
            end_date = end_date if end_datetime > start_datetime else None

    try:
        data = pdr.get_data_yahoo(ticker,
                                  start=start_date,
                                  end=end_date)
    except KeyError as e:
        logger.warning(f' > It was not possible to load the data for {ticker}')
        data = None

    return data

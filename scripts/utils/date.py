import datetime
import time
from typing import Text

import pandas as pd


def date_conversion_to_pandas_date_str(date: Text):
    parsed_date = pd.to_datetime(date, dayfirst=True)
    str_date = str(parsed_date.date())

    return str_date


def readable_timestamp():
    timestamp = time.time()
    date_obj = datetime.datetime.fromtimestamp(timestamp)

    return date_obj.strftime('%Y-%m-%d_%H-%M-%S')


def readable_date():
    timestamp = readable_timestamp()

    return timestamp.split('_')[0]


def today() -> Text:
    return datetime.date.today()


def yesterday() -> Text:
    return today() - datetime.timedelta(days=1)


def parse_date(date_str: Text):
    date_format = '%Y-%m-%d'
    return datetime.datetime.strptime(date_str, date_format)

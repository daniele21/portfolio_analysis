import os
from typing import Text

import pandas as pd

from portfolio_analysis.scripts.data.load import load_csv
from portfolio_analysis.scripts.data.yahoo_extraction import extract_data
from portfolio_analysis.scripts.utils.date import yesterday, check_date
from portfolio_analysis.scripts.utils.logging import setup_logger
from portfolio_analysis.scripts.utils.pandas_memory import pandas_series_to_float32


class Ticker:

    def __init__(self,
                 ticker_id: Text,
                 folder_dir: Text,
                 name: Text = None,
                 isin: Text = None,
                 instrument: Text = None,
                 risk: int = None,
                 fee: float = None):
        self.id = ticker_id

        self.name = name
        self.isin = isin
        self.instrument = instrument
        self.risk = risk
        self.fee = fee
        self.path = f'{folder_dir}/{ticker_id}.csv'
        self.logger = setup_logger(f'{ticker_id}')

        self.data = self._init_data()

    def _init_data(self) -> pd.DataFrame:

        if os.path.exists(self.path):
            data = load_csv(self.path)
        else:
            yesterday_date = str(yesterday())
            data = self._load_data(start_date=None, end_date=yesterday_date)
            if data is None:
                return None

            self.logger.info(f' > Saving ticker data at {self.path}')
            data.to_csv(self.path)

        for col in data.columns:
            data[col] = pandas_series_to_float32(data[col])
            # data[col] = data[col][data[col].duplicated() == False]

        return data

    def _load_data(self, start_date, end_date=None):
        data = extract_data(ticker=self.id,
                            start_date=start_date,
                            end_date=end_date)
        return data

    def get_data_from_date(self,
                           start_date: Text,
                           end_date: Text):
        check = check_date(start_date) and check_date(end_date)
        if not check:
            return None

        data = self.data.set_index('Date') if 'Date' in self.data.columns else self.data
        filter_df = data[start_date: end_date].reset_index()
        filter_df = filter_df.drop(filter_df[filter_df.duplicated()].index)
        filter_df = filter_df.set_index('Date') if 'Date' in filter_df.columns else filter_df

        return filter_df

    def update_data(self):
        import datetime as dt

        if len(self.data) == 0:
            return

        last_date = str((self.data.index.to_list()[-1]).date())
        yesterday_date = str(yesterday())

        last_date_dt, yest_dt = dt.datetime.strptime(last_date, '%Y-%m-%d'), dt.datetime.strptime(yesterday_date,
                                                                                                  '%Y-%m-%d')

        if last_date_dt < yest_dt:
            start_date = last_date_dt + dt.timedelta(1)
            self.logger.info(f' > Updating {self.id} data from {str(last_date_dt.date())} to {str(yest_dt.date())}')
            update_data = self._load_data(start_date=start_date,
                                          end_date=yesterday_date)

            self.data = self.data.append(update_data)
            self.data = self.data.reset_index() \
                .drop_duplicates(subset='Date', keep='last') \
                .set_index('Date').sort_index()
            self.save()

        return

    def save(self):
        self.logger.info(f' > Saving ticker data at {self.path}')
        self.data.to_csv(self.path)

        return

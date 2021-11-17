from threading import Thread
from typing import Text

from scripts.data.load import load_csv
from scripts.portfolio.ticker import Ticker
import pandas as pd


class Tickers:

    def __init__(self,
                 ticker_details_path: Text,
                 ticker_data_folder: Text):
        self.ticker_details_path = ticker_details_path
        self.ticker_data_folder = ticker_data_folder

        self.ticker_details_df = load_csv(ticker_details_path)
        self.tickers_dict = {}

        self._init_data()

    def _init_data(self):
        ticker_list = self.ticker_details_df['ticker_id'].to_list()

        for i, ticker_id in enumerate(ticker_list):
            ticker_detail_dict = self.ticker_details_df[self.ticker_details_df['ticker_id'] == ticker_id].to_dict()

            ticker = Ticker(ticker_id=ticker_id,
                            folder_dir=self.ticker_data_folder,
                            name=ticker_detail_dict['ticker_name'][i],
                            isin=ticker_detail_dict['isin'][i],
                            instrument=ticker_detail_dict['instrument'][i],
                            risk=ticker_detail_dict['risk'][i],
                            fee=ticker_detail_dict['fee'][i],
                            )

            self.tickers_dict[ticker_id] = ticker

    def add_ticker(self,
                   ticker_id: Text,
                   name: Text,
                   isin: Text,
                   instrument: Text,
                   risk: int,
                   fee: float):

        ticker = Ticker(ticker_id=ticker_id,
                        folder_dir=self.ticker_data_folder,
                        name=name,
                        isin=isin,
                        instrument=instrument,
                        risk=risk,
                        fee=fee
                        )

        ticker_df = self.ticker_details_df[self.ticker_details_df['ticker_id'] == ticker_id]
        if len(ticker_df) > 0:
            self.ticker_details_df = self.ticker_details_df.drop(
                self.ticker_details_df[self.ticker_details_df['ticker_id'] == ticker_id].index)

        self.ticker_details_df = self.ticker_details_df.append(pd.DataFrame({'ticker_id': ticker_id,
                                                                             'ticker_name': name,
                                                                             'isin': isin,
                                                                             'instrument': instrument,
                                                                             'risk': risk,
                                                                             'fee': fee}, index=[0]))
        self.ticker_details_df = self.ticker_details_df.reset_index(drop=True)
        self.save_details()

        self.tickers_dict[ticker_id] = ticker

        return

    def update_tickers_data(self):
        thread_list = []
        for ticker_id in self.tickers_dict:
            ticker = self.tickers_dict[ticker_id]
            thread = Thread(target=ticker.update_data)
            thread.start()
            thread_list.append(thread)

        for t in thread_list:
            t.join()

        return

    def get_ticker(self,
                   ticker_id: Text) -> Ticker:
        ticker_df = self.tickers_dict[ticker_id]
        ticker_df.data['Date'] = pd.to_datetime(ticker_df.data.index)
        ticker_df.data = ticker_df.data.reset_index(drop=True)
        return ticker_df

    def get_ticker_details(self,
                           ticker_id: Text):
        return self.ticker_details_df[self.ticker_details_df['ticker_id'] == ticker_id].iloc[0].to_dict()

    def get_ticker_ids(self):
        return list(self.tickers_dict.keys())

    def get_tickers_by_instrument(self,
                                  instrument: Text):
        assert instrument in ['ETF', 'Crypto', 'Stock'], f' > No valid instrument: {instrument}'
        instrument_df = self.ticker_details_df[self.ticker_details_df['instrument'] == instrument]

        return instrument_df

    def save_details(self):
        self.ticker_details_df.to_csv(self.ticker_details_path)


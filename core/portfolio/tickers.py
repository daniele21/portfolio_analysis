import logging
from copy import deepcopy
from threading import Thread
from typing import Text, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.operations.optimizations import get_efficient_frontier, msr, optimal_weights
from core.operations.time_series import sharpe_ratio, get_df_from_dict, get_cov, portfolio_vol, annualized_rets, \
    portfolio_return, get_er
from core.portfolio.ticker import Ticker
from scripts.constants.constants import RISK_FREE_RATE
from scripts.data.load import load_ticker_details
from scripts.visualization.optimization import optimization_plot

logger = logging.getLogger('Tickers')


class Tickers:

    def __init__(self,
                 ticker_details_path: Text,
                 ticker_data_folder: Text):
        self.ticker_details_path = ticker_details_path
        self.ticker_data_folder = ticker_data_folder

        self.ticker_details_df = load_ticker_details(ticker_details_path)
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

    def get_tickers_dict(self,
                         features: List = None,
                         start_date: Text = None):
        tickers_dict = {}

        for ticker in tqdm(self.tickers_dict):
            ticker_df = self.tickers_dict[ticker].data
            if len(ticker_df) > 1 and (features is None or (features is not None and ticker in features)):
                tickers_dict[ticker] = ticker_df['Close'] if start_date is None else ticker_df[start_date:]['Close']

        return tickers_dict

    def get_tickers_return_dict(self,
                                start_date: Text = None,
                                features: List = None,
                                freq: str = 'W'):
        # if freq not in ['W', 'M']:
        #     logger.warning(f' > Frequency is not W nor M')

        tickers_dict = self.get_tickers_dict(features, start_date)

        returns_dict = {}
        for ticker in tickers_dict:
            returns_dict[ticker] = tickers_dict[ticker].pct_change(freq=freq).dropna()

        return returns_dict

    def get_sharpe_ratios(self,
                          freq: str,
                          periods_per_year: int,
                          features: List = None,
                          start_date: Text = None,
                          risk_free_rate: float = RISK_FREE_RATE,
                          ):
        return_df = self.get_tickers_return_dict(start_date=start_date,
                                                 features=features,
                                                 freq=freq)
        sharpe_ratios = {}

        for ticker in return_df:
            ticker_df = return_df[ticker]
            ticker_df = ticker_df.dropna()
            if len(ticker_df) > 1:
                sharpe_ratios[ticker] = sharpe_ratio(ticker_df.dropna(),
                                                     risk_free_rate=risk_free_rate,
                                                     periods_per_year=periods_per_year)

        return sharpe_ratios

    def get_tickers_volatility(self,
                               freq: str,
                               features: List = None,
                               start_date: Text = None,
                               weights=None,
                               ):
        returns_dict = self.get_tickers_return_dict(start_date=start_date,
                                                    features=features,
                                                    freq=freq)

        returns_df = get_df_from_dict(returns_dict)
        cov = get_cov(returns_df)
        tickers = cov.columns

        weights = np.repeat(1 / len(tickers), len(tickers)) if weights is None else weights
        volatility = portfolio_vol(weights, cov.loc[tickers, tickers])

        return volatility

    def get_tickers_return(self,
                           freq: str,
                           periods_per_year: int,
                           features: List = None,
                           start_date: Text = None,
                           weights=None):
        returns_dict = self.get_tickers_return_dict(start_date=start_date,
                                                    features=features,
                                                    freq=freq)

        returns_df = get_df_from_dict(returns_dict)
        returns = annualized_rets(returns_df, periods_per_year)

        weights = np.repeat(1 / len(returns_df.columns), len(returns_df.columns)) if weights is None else weights
        port_return = portfolio_return(returns[returns_df.columns], weights)

        return port_return

    def get_efficient_frontier(self,
                               n_points,
                               freq: str,
                               periods_per_year: int,
                               features: List = None,
                               start_date: Text = None
                               ):

        returns_dict = self.get_tickers_return_dict(start_date=start_date,
                                                    features=features,
                                                    freq=freq)
        returns_df = get_df_from_dict(returns_dict)
        er = get_er(returns_df, periods_per_year)
        sub_returns_df = returns_df[er.index].dropna()
        cov = get_cov(sub_returns_df)

        ef = get_efficient_frontier(n_points, er, cov)

        return ef, er, cov

    def plot_efficient_frontier(self,
                                n_points,
                                freq: str,
                                periods_per_year: int,
                                features: List = None,
                                start_date: Text = None):

        ef, er, cov = self.get_efficient_frontier(n_points, freq, periods_per_year,
                                                  features, start_date)

        optimization_plot(ef, er, cov, title='Portfolio Optimization')

        return ef, er, cov

    def get_max_sharpe_ratio(self,
                             risk_free_rate: float = RISK_FREE_RATE,
                             start_date: Text = None,
                             periods_per_year=252):
        returns_dict = self.get_tickers_return_dict()
        returns_df = get_df_from_dict(returns_dict)
        returns_df = returns_df[start_date:] if start_date is not None else returns_df
        cov = returns_df[start_date:].cov() if start_date is not None else returns_df.cov()
        er = annualized_rets(returns_df, periods_per_year)

        er = er[returns_df.columns]
        cov = cov.loc[returns_df.columns, returns_df.columns]

        msr_value = msr(risk_free_rate, er, cov)
        msr_weights = pd.DataFrame(msr_value, index=returns_df.columns)

        vol = self.get_tickers_volatility(weights=msr_value)
        ret = self.get_tickers_return(weights=msr_value)

        return msr_weights, vol, ret

    def get_portfolio_min_vol(self,
                              start_date: Text = None,
                              periods_per_year=252,
                              n_points: int = 20,
                              ):
        returns_dict = self.get_tickers_return_dict()
        returns_df = get_df_from_dict(returns_dict)
        returns_df = returns_df[start_date:] if start_date is not None else returns_df
        cov = returns_df[start_date:].cov() if start_date is not None else returns_df.cov()
        er = annualized_rets(returns_df, periods_per_year)

        er = er[returns_df.columns]
        cov = cov.loc[returns_df.columns, returns_df.columns]
        weights = optimal_weights(n_points, er, cov)
        rets = [portfolio_return(w, er) for w in weights]
        vol = [portfolio_vol(w, cov) for w in weights]
        rv = pd.DataFrame({
            "returns": rets,
            "volatility": vol,
            "weights": weights
        })
        rv = rv.sort_values(by='volatility')
        result = rv.iloc[0]

        return result

    def get_ticker(self,
                   ticker_id: Text) -> Ticker:
        ticker = self.tickers_dict[ticker_id]
        # ticker.data['Date'] = pd.to_datetime(ticker.data.index)
        # ticker.data = ticker.data.reset_index(drop=True)

        return deepcopy(ticker)

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

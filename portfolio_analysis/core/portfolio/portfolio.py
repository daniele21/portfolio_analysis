import os
from typing import Text

import pandas as pd

from portfolio_analysis.core.portfolio.ticker import Ticker
from portfolio_analysis.core.portfolio.tickers import Tickers
from portfolio_analysis.scripts.data.load import load_portfolio_transactions
from portfolio_analysis.scripts.utils.date import today
from portfolio_analysis.scripts.utils.logging import setup_logger

logger = setup_logger('Portfolio')


class Portfolio:

    def __init__(self,
                 transactions_path: Text):
        self.transactions_path = transactions_path
        self.transactions = self._init_transactions()

    def _init_transactions(self):
        logger.info(' > Init Portfolio transactions')
        if os.path.exists(self.transactions_path):
            return load_portfolio_transactions(self.transactions_path)
        else:
            return pd.DataFrame(columns=['date', 'ticker_id', 'action', 'quantity',
                                         'price', 'commission', 'gain', 'deposit', 'spent'])

    def get_amount_spent(self,
                         instrument: Text = None,
                         tickers: Tickers = None):
        if instrument is not None:
            ticker_ids = tickers.get_tickers_by_instrument(instrument)
            amount_spent = 0
            for ticker_id in ticker_ids:
                ticker_df = self.transactions[self.transactions['ticker_id'] == ticker_id]
                amount_spent += ticker_df['spent'].sum() - ticker_df['gain'].sum()

        else:
            amount_spent = self.transactions['spent'].sum() - self.transactions['gain'].sum()

        return amount_spent

    def get_ticker_performance(self,
                               ticker: Ticker):
        logger.debug(f' > Computing ticker performance for ticker: {ticker.id}')

        ticker_transactions = self.transactions[self.transactions['ticker_id'] == ticker.id]
        if len(ticker_transactions) == 0:
            logger.error(f' > No transaction of ticker {ticker.id} found!')
            return None

        dates = ticker_transactions['date'].to_list()

        ticker_performance = pd.DataFrame()

        i, cum_quantity, cum_spent = 0, 0, 0
        for index, row in ticker_transactions.iterrows():
            action = row['action']
            quantity = row['quantity']
            price = row['price']
            commission = row['commission']

            if action == 'buy':
                spent = (quantity * price) + commission if action == 'buy' else None
                cum_quantity += quantity

            elif action == 'sell':
                cum_quantity -= quantity
                spent = ((quantity * price) - commission) * (-1)

            else:
                raise Exception(f'No valid action: {action} for ticker {ticker.id}')

            cum_spent += spent

            start_date = dates[i]
            end_date = dates[i + 1] if i + 1 < len(dates) else str(today())
            i += 1

            ticker_df = ticker.get_data_from_date(start_date=start_date,
                                                  end_date=end_date).iloc[1:]
            output_dict = {'ticker_id': [ticker.id] * len(ticker_df),
                           'price': ticker_df['Close'],
                           'volume': ticker_df['Volume'],
                           'cum_quantity': [cum_quantity] * len(ticker_df),
                           'cum_spent': [cum_spent] * len(ticker_df)
                           }
            output_df = pd.DataFrame(output_dict, index=ticker_df.index)
            output_df['potential_gain'] = (output_df['price'] * output_df['cum_quantity']) - commission - output_df[
                'cum_spent']
            output_df['performance'] = output_df['potential_gain'] / output_df['cum_spent']

            ticker_performance = ticker_performance.append(output_df)

        return ticker_performance

    def get_portfolio_performance(self, tickers: Tickers):

        logger.debug(f' > Computing the portfolio performance')

        portfolio_performance = pd.DataFrame()
        cum_spent_cols, pot_gain_cols = [], []

        for ticker in tickers.tickers_dict.values():
            ticker_performance = self.get_ticker_performance(ticker)

            cum_spent_col, pot_gain_col = f'{ticker.id}_cum_spent', f'{ticker.id}_potential_gain'
            cum_spent_cols.append(cum_spent_col)
            pot_gain_cols.append(pot_gain_col)
            ticker_df = pd.DataFrame({cum_spent_col: ticker_performance['cum_spent'].to_list(),
                                      pot_gain_col: ticker_performance['potential_gain'].to_list()},
                                     index=ticker_performance.index)
            portfolio_performance = pd.concat((portfolio_performance, ticker_df), axis=1)
            portfolio_performance = portfolio_performance.drop_duplicates()

        portfolio_performance = portfolio_performance.fillna(method='ffill')
        portfolio_performance['cum_spent'] = portfolio_performance[cum_spent_cols].sum(axis=1)
        portfolio_performance['potential_gain'] = portfolio_performance[pot_gain_cols].sum(axis=1)

        portfolio_performance['performance'] = portfolio_performance['potential_gain'] / portfolio_performance['cum_spent']

        return portfolio_performance

    def get_group_performances(self, tickers: Tickers):

        logger.debug(f' > Computing performance for group of tickers: {tickers.instruments}')

        group_performances = {}

        for instr in tickers.instruments:
            ticker_id_list = tickers.get_tickers_by_instrument(instr)
            group_performance = pd.DataFrame()
            cum_spent_cols, pot_gain_cols = [], []

            for ticker_id in ticker_id_list:
                ticker = tickers.tickers_dict[ticker_id]

                ticker_performance = self.get_ticker_performance(ticker)
                # group_performance = group_performance.append(ticker_performance)
                cum_spent_col, pot_gain_col = f'{ticker_id}_cum_spent', f'{ticker_id}_potential_gain'
                cum_spent_cols.append(cum_spent_col)
                pot_gain_cols.append(pot_gain_col)
                ticker_df = pd.DataFrame({cum_spent_col: ticker_performance['cum_spent'].to_list(),
                                          pot_gain_col: ticker_performance['potential_gain'].to_list()},
                                         index=ticker_performance.index)
                group_performance = pd.concat((group_performance, ticker_df), axis=1)

            group_performance = group_performance.fillna(method='ffill')
            group_performance['cum_spent'] = group_performance[cum_spent_cols].sum(axis=1)
            group_performance['potential_gain'] = group_performance[pot_gain_cols].sum(axis=1)

            group_performance['performance'] = group_performance['potential_gain'] / group_performance['cum_spent']

            group_performances[instr] = group_performance

        return group_performances

    def get_ticker_transactions(self,
                                ticker_id: Text):
        ticker_df = self.transactions[self.transactions['ticker_id'] == ticker_id]
        return ticker_df

    def get_actual_stake(self,
                         tickers: Tickers):
        df = tickers.ticker_details_df[['ticker_id', 'instrument']]
        amount_spent = self.get_amount_spent()
        ticker_stake = {x: 0.0 for x in tickers.instruments}

        for i, row in df.iterrows():
            ticker_id, instrument = row['ticker_id'], row['instrument']
            ticker_df = self.get_ticker_transactions(ticker_id)
            ticker_spent = ticker_df['spent'].sum() - ticker_df['gain'].sum()
            ticker_stake[instrument] += ticker_spent / amount_spent

        return ticker_stake

    def get_actual_stake_by_instrument(self,
                                       instrument: Text,
                                       tickers: Tickers,
                                       ):
        amount_spent = self.get_amount_spent(instrument=instrument,
                                             tickers=tickers)
        tickers_ids = tickers.get_tickers_by_instrument(instrument)
        instrument_stake = {ticker_id: 0 for ticker_id in tickers_ids}

        for ticker_id in tickers_ids:
            ticker_df = self.transactions[self.transactions['ticker_id'] == ticker_id]
            instrument_stake[ticker_id] += (ticker_df['spent'].sum() - ticker_df['gain'].sum()) / amount_spent

        return instrument_stake

    def get_actual_stake_by_risk(self, tickers: Tickers):
        amount_spent = self.get_amount_spent(tickers=tickers)
        ticker_items = tickers.ticker_details_df
        etf_risk_stake = {str(int(risk)): 0 for risk in ticker_items['risk'].to_list()}

        for ticker_id in ticker_items['ticker_id'].to_list():
            ticker_df = self.transactions[self.transactions['ticker_id'] == ticker_id]
            risk = ticker_items[ticker_items['ticker_id'] == ticker_id]['risk'].iloc[0]
            etf_risk_stake[str(int(risk))] += (ticker_df['spent'].sum() - ticker_df['gain'].sum()) / amount_spent

        return etf_risk_stake

    def save_transactions(self):
        self.transactions.to_csv(self.transactions_path)
        logger.info(f' > Saving transactions at {self.transactions_path}')

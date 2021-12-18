import logging
import os
import re
from typing import Text
import pandas as pd

from scripts.constants.constants import ETF, CRYPTO, STOCK
from scripts.data.load import load_csv
from scripts.portfolio.ticker import Ticker
from scripts.portfolio.tickers import Tickers
from scripts.utils.date import today

logger = logging.getLogger('Portfolio')


class Portfolio:

    def __init__(self,
                 transactions_path: Text):
        self.transactions_path = transactions_path
        self.transactions = self._init_transactions()

    def _init_transactions(self):
        if os.path.exists(self.transactions_path):
            return load_csv(self.transactions_path)
        else:
            return pd.DataFrame(columns=['date', 'ticker_id', 'action', 'quantity',
                                         'price', 'commission', 'gain', 'deposit'])

    def add_transaction(self,
                        tickers: Tickers,
                        date: Text,
                        ticker_id: Text,
                        action: Text,
                        quantity: float,
                        price: float,
                        commission: float,
                        gain: float,
                        deposit: float):

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

        if action != 'deposit' and ticker_id not in tickers.tickers_dict:
            logger.error(f' > Ticker {ticker_id} not in tickers details')
            return

        row = {'date': date,
               'ticker_id': ticker_id,
               'action': action,
               'quantity': quantity,
               'price': price,
               'commission': commission,
               'gain': gain,
               'deposit': deposit,
               'spent': (quantity * price) + commission}

        self.transactions = self.transactions.append(pd.DataFrame(row, index=[0])) \
            .reset_index(drop=True)

        logger.info(f' > Ticker {ticker_id} transaction inserted at {self.transactions}')
        self.save_transactions()

    def get_amount_spent(self,
                         instrument: Text = None,
                         tickers: Tickers = None):
        if instrument is not None:
            instrument_df = tickers.get_tickers_by_instrument(instrument)
            ticker_ids = instrument_df['ticker_id'].to_list()
            amount_spent = 0
            for ticker_id in ticker_ids:
                ticker_df = self.transactions[self.transactions['ticker_id'] == ticker_id]
                amount_spent += ticker_df['spent'].sum() - ticker_df['gain'].sum()

        else:
            amount_spent = self.transactions['spent'].sum() - self.transactions['gain'].sum()

        return amount_spent

    def get_ticker_performance(self,
                               ticker: Ticker):
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

    def get_ticker_performances(self,
                                tickers: Tickers):
        performances = pd.DataFrame()

        for ticker_id in tickers.tickers_dict:
            ticker_performance_df = self.get_ticker_performance(tickers.get_ticker(ticker_id))
            performances = performances.merge(ticker_performance_df['performance'], right_index=True)
            performances = performances.rename(columns={'performance': ticker_id})

    def get_ticker_transactions(self,
                                ticker_id: Text):
        ticker_df = self.transactions[self.transactions['ticker_id'] == ticker_id]
        return ticker_df

    def get_actual_stake(self,
                         tickers: Tickers):
        df = tickers.ticker_details_df[['ticker_id', 'instrument']]
        amount_spent = self.get_amount_spent()
        ticker_stake = {ETF: 0,
                        CRYPTO: 0,
                        STOCK: 0}

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
        instrument_stake = {ticker_id: 0 for ticker_id in tickers_ids['ticker_id'].to_list()}

        for ticker_id in tickers_ids['ticker_id'].to_list():
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

import os
import unittest

from scripts.paths import TRANSACTION_PATH, TICKER_DETAILS_PATH, TICKER_DATA_DIR
from scripts.portfolio.portfolio import Portfolio
from scripts.portfolio.tickers import Tickers


class TestPortfolio(unittest.TestCase):
    os.chdir('../')
    tickers = Tickers(TICKER_DETAILS_PATH, TICKER_DATA_DIR)
    portfolio = Portfolio(TRANSACTION_PATH)

    def test_init_portfolio(self):
        portfolio = Portfolio(TRANSACTION_PATH)

        self.assertIsNotNone(portfolio)

    def test_retrieve_ticker_performance(self):
        ticker = self.tickers.tickers_dict['WTEC.MI']
        portfolio = Portfolio(TRANSACTION_PATH)
        performance = portfolio.get_ticker_performance(ticker)

        self.assertIsNotNone(performance)

    def test_add_transactions(self):
        ticker_id = 'ETH-EUR'
        tickers = self.tickers
        date = '2021-11-16'
        action = 'buy'
        quantity = 0.0379
        price = 3719
        commission = 9.05
        gain = None
        deposit = None

        portfolio = Portfolio(TRANSACTION_PATH)
        portfolio.add_transaction(tickers=tickers,
                                  date=date,
                                  ticker_id=ticker_id,
                                  action=action,
                                  quantity=quantity,
                                  price=price,
                                  commission=commission,
                                  gain=gain,
                                  deposit=deposit)

    def test_get_amount_spent(self):
        amount_spent = self.portfolio.get_amount_spent()
        etf_amount_spent = self.portfolio.get_amount_spent('ETF', self.tickers)
        crypto_amount_spent = self.portfolio.get_amount_spent('Crypto', self.tickers)
        stock_amount_spent = self.portfolio.get_amount_spent('Stock', self.tickers)

        return

    def test_actual_stake(self):
        portfolio = Portfolio(TRANSACTION_PATH)
        stake = portfolio.get_actual_stake(self.tickers)

        self.assertIsNotNone(stake)

    def test_actual_stake_by_instrument(self):
        portfolio = Portfolio(TRANSACTION_PATH)
        etf_stake = portfolio.get_actual_stake_by_instrument(instrument='ETF',
                                                             tickers=self.tickers)

        crypto_stake = portfolio.get_actual_stake_by_instrument(instrument='Crypto',
                                                                tickers=self.tickers)

        stock_stake = portfolio.get_actual_stake_by_instrument(instrument='Stock',
                                                               tickers=self.tickers)

        return

    def test_actual_stake_by_etf(self):
        portfolio = Portfolio(TRANSACTION_PATH)
        portfolio.get_actual_stake_by_etf(tickers=self.tickers)

        return

    def test_ticker_performances(self):
        portfolio = Portfolio(TRANSACTION_PATH)
        portfolio.get_ticker_performances(tickers=self.tickers)


if __name__ == '__main__':
    unittest.main()

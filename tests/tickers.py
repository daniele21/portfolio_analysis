import unittest

from scripts.portfolio.tickers import Tickers
from scripts.paths import TICKER_DATA_DIR, TICKER_DETAILS_PATH


class TestTickers(unittest.TestCase):
    def test_tickers_init(self):
        tickers = Tickers(TICKER_DETAILS_PATH, TICKER_DATA_DIR)

        self.assertIsNotNone(tickers)

    def test_tickers_update_ticker(self):
        tickers = Tickers(TICKER_DETAILS_PATH, TICKER_DATA_DIR)
        tickers.update_tickers_data()

    def test_add_ticker(self):
        tickers = Tickers(TICKER_DETAILS_PATH, TICKER_DATA_DIR)

        ticker_id = 'FET-EUR'
        ticker_name = 'Fetch-AI'

        tickers.add_ticker(ticker_id=ticker_id,
                           name=ticker_name,
                           isin=None,
                           instrument='Crypto',
                           risk=7,
                           fee=0.0)


if __name__ == '__main__':
    unittest.main()

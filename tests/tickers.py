import unittest

from core.portfolio.tickers import Tickers
from scripts.constants.paths import TICKER_DATA_DIR, TICKER_DETAILS_PATH


class TestTickers(unittest.TestCase):
    tickers = Tickers(TICKER_DETAILS_PATH, TICKER_DATA_DIR)

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

    def test_get_tickers_df(self):
        features = None
        start_date = None

        ticker_dict = self.tickers.get_tickers_dict(features, start_date)

        self.assertIsNotNone(ticker_dict)

    def test_get_tickers_return_df(self):
        features = None
        start_date = None
        freq = 'M'

        returns_dict = self.tickers.get_tickers_return_dict(features, start_date, freq)

        self.assertIsNotNone(returns_dict)

    def test_get_sharpe_ratio(self):
        # start_date = '2021'
        features = None
        start_date = None
        freq = 'M'
        periods_per_year = 12

        sharpe_ratios = self.tickers.get_sharpe_ratios(start_date=start_date,
                                                       freq=freq,
                                                       periods_per_year=periods_per_year,
                                                       features=features,
                                                       )

        self.assertIsNotNone(sharpe_ratios)

    def test_get_tickers_volatility(self):
        features = None
        start_date = None
        freq = 'M'
        periods_per_year = 12

        volatility = self.tickers.get_tickers_volatility(start_date=start_date,
                                                         freq=freq,
                                                         features=features,
                                                         )

        self.assertIsNotNone(volatility)

    def test_get_tickers_return(self):
        features = None
        start_date = None
        freq = 'M'
        periods_per_year = 12

        port_return = self.tickers.get_tickers_return(start_date=start_date,
                                                      freq=freq,
                                                      periods_per_year=periods_per_year,
                                                      features=features,
                                                      )

        self.assertIsNotNone(port_return)

    def test_plot_efficient_frontier(self):
        start_date = None
        points = 30
        features = ['AIAI.MI', 'BATT.MI', 'USPY.DE', 'SWDA.MI',
                    'ESPO.MI']
        # freq = 'D'
        # periods_per_years = 252
        freq = 'W'
        periods_per_years = 52
        freq = 'M'
        periods_per_years = 12
        features = None

        ef, er, cov = self.tickers.plot_efficient_frontier(points,
                                                           features=features,
                                                           freq=freq,
                                                           periods_per_year=periods_per_years,
                                                           start_date=start_date)

        self.assertIsNotNone(ef)

    def test_get_max_sharpe_ratio(self):
        start_date = '2021'

        msr, vol, ret = self.tickers.get_max_sharpe_ratio(start_date=start_date)
        self.tickers.plot_efficient_frontier(20)
        self.assertIsNotNone(msr)

    def test_get_portfolio_min_vol(self):
        start_date = None

        result = self.tickers.get_portfolio_min_vol(start_date=start_date)
        self.tickers.plot_efficient_frontier(20)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()

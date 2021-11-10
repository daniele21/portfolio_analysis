import unittest

from scripts.data.ticker import add_ticker_data, add_ticker_details
from scripts.data.transaction import add_transaction


class DataTests(unittest.TestCase):
    def test_add_ticker_data(self):
        ticker = 'AIAI.MI'
        ticker_df = add_ticker_data(ticker=ticker)

        self.assertIsNotNone(ticker_df)

    def test_add_ticker_detail(self):
        ticker_id = "AIAI.MI"
        ticker_name = "L&G Artificial Intelligence UCITS ETF"
        isin = "IE00BK5BCD43"
        instrument = "ETF"
        risk = 6
        fee = 0.49

        ticker_details = add_ticker_details(ticker_id=ticker_id,
                                            ticker_name=ticker_name,
                                            isin=isin,
                                            instrument=instrument,
                                            risk=risk,
                                            fee=fee, )

        self.assertIsNotNone(ticker_details)

    def test_add_transaction(self):
        date = "2021-04-23"
        ticker_id = None
        action = "deposit"
        quantity = 0
        price = 0
        commission = 0
        gain = None
        deposit = 500.00

        transaction_df = add_transaction(date=date,
                                         ticker_id=ticker_id,
                                         action=action,
                                         quantity=quantity,
                                         price=price,
                                         commission=commission,
                                         gain=gain,
                                         deposit=deposit)

        self.assertIsNotNone(transaction_df)


if __name__ == '__main__':
    unittest.main()

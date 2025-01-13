import pandas as pd

class Transaction:
    """
    Represents a single transaction: buy or sell.
    """

    def __init__(self,
                 operation: str,  # "Buy" or "Sell"
                 date_: pd.Timestamp,
                 valuta: pd.Timestamp,
                 ticker_symbol: str,
                 quantity: float):
        self.operation = operation.lower()  # e.g. "buy" or "sell"
        self.date = date_
        # self.valuta = valuta
        self.ticker_symbol = ticker_symbol.upper()
        self.quantity = quantity



    def __repr__(self):
        return (f"<Transaction {self.operation.upper()} {self.quantity} of "
                f"{self.ticker_symbol} on {self.date.strftime('%Y-%m-%d')}>")

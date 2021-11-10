import json
import logging
from typing import Text
import pandas as pd
from scripts.data.ticker import add_ticker_details
from scripts.data.transaction import add_transaction
from scripts.paths import TRANSACTION_JSON_PATH, TRANSACTION_PATH, TICKER_DETAILS_PATH

logger = logging.getLogger('Setup')


def setup_ticker_details(ticker_details_json_path: Text):
    with open(ticker_details_json_path, 'r') as j:
        ticker_details = json.load(j)
        j.close()

    # Clear ticker_details
    ticker_details_df = pd.DataFrame()
    ticker_details_df.to_csv(TICKER_DETAILS_PATH)

    for item in ticker_details:
        ticker_id = item['ticker_id']
        ticker_name = item['ticker_name']
        isin = item['isin']
        instrument = item['instrument']
        risk = item['risk']
        fee = item['fee']

        ticker_details_df = add_ticker_details(ticker_id=ticker_id,
                                               ticker_name=ticker_name,
                                               isin=isin,
                                               instrument=instrument,
                                               risk=risk,
                                               fee=fee
                                               )
        logger.info(f' > Added info ticker about {ticker_id}')

    return ticker_details_df


def setup_transactions(transactions_json_path: Text):
    with open(transactions_json_path, 'r') as j:
        transactions = json.load(j)
        j.close()

    # Clear transactions
    transactions_df = pd.DataFrame()
    transactions_df.to_csv(TRANSACTION_PATH)

    for t in transactions:
        date = t["date"]
        ticker_id = t["ticker_id"]
        action = t["action"]
        quantity = t["quantity"]
        price = t["price"]
        commission = t["commission"]
        gain = t["gain"]
        deposit = t["deposit"]

        transactions_df = add_transaction(date=date,
                                          ticker_id=ticker_id,
                                          action=action,
                                          quantity=quantity,
                                          price=price,
                                          commission=commission,
                                          gain=gain,
                                          deposit=deposit)

    return transactions_df


if __name__ == '__main__':
    ticker_details_path = 'resources/tickers.json'
    transaction_path = TRANSACTION_JSON_PATH

    # ticker_details = setup_ticker_details(ticker_details_path)
    transactions = setup_transactions(transaction_path)
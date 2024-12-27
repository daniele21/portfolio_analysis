# config.py
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TICKER_DATA_DIR = os.path.join(DATA_DIR, 'ticker_data')

TICKERS_PATH = os.path.join(BASE_DIR, 'tickers_test.json')          # Sostituisci con il percorso corretto
TRANSACTIONS_PATH = os.path.join(BASE_DIR, 'transactions_test.json')  # Sostituisci con il percorso corretto

# Assicurati che le directory esistano
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TICKER_DATA_DIR, exist_ok=True)

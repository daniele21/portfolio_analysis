from datetime import date
from typing import Dict, List, Optional, Any

import pandas as pd
import yfinance as yf


class Ticker:
    """
    Handles a single symbol's historical price data and fundamentals.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.data: pd.DataFrame = pd.DataFrame()  # Will store OHLC data
        self.fundamentals: Dict[str, Any] = {}  # Will store key fundamentals
        self.title: str = self.symbol  # Human-readable name/longName if available
        self.asset_type: Optional[str] = None  # e.g. "EQUITY", "ETF", "MUTUALFUND"
        self.sector: Optional[str] = None  # from fundamentals
        self.industry: Optional[str] = None  # from fundamentals
        self.category: Optional[str] = None  # from fundamentals

    def fetch_fundamental_data(self):
        """
        Fetch fundamental data (financials, sector, country, etc.) via yfinance.
        """
        try:
            stock = yf.Ticker(self.symbol)
            info = stock.info  # Deprecated in yfinance >= 0.2.4, but still works for now

            self.fundamentals = {
                "Quote Type": info.get("quoteType"),
                "Category": info.get('category'),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "Full Name": info.get("longName"),
            }

            # Update self.title if we have a longName
            if self.fundamentals.get("Full Name"):
                self.title = self.fundamentals["Full Name"]
            if self.fundamentals.get("Category"):
                self.category = self.fundamentals["Category"]
            if self.fundamentals.get("Quote Type"):
                self.asset_type = self.fundamentals["Quote Type"]  # e.g., 'EQUITY'
            if self.fundamentals.get("Sector"):
                self.sector = self.fundamentals["Sector"]
            if self.fundamentals.get("Industry"):
                self.industry = self.fundamentals["Industry"]

        except Exception as e:
            self.fundamentals = {"Error": str(e)}

    def fetch_price_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch and store historical OHLC (Open, High, Low, Close, Adj Close, Volume) data
        between start_date and end_date. Reindex to business-day frequency and forward-fill missing data.
        """
        try:
            df = yf.download(
                self.symbol,
                start=start_date,
                end=end_date,
                progress=False,
                group_by="ticker"
            )

            if df.empty:
                raise ValueError(f"No data returned by yfinance for {self.symbol}.")

            # If we have a MultiIndex, flatten it (e.g., ("AAPL", "Close") -> "AAPL_Close")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(col).strip() for col in df.columns.values]
            else:
                # Otherwise, just title-case them (e.g. 'open' -> 'Open')
                df.columns = [col.title() for col in df.columns]

            # Map any columns that look like SYMBOL_Close -> 'Close'
            possible_mappings = {}
            for col in df.columns:
                lc = col.lower()
                if lc.endswith("_open"):
                    possible_mappings[col] = "Open"
                elif lc.endswith("_high"):
                    possible_mappings[col] = "High"
                elif lc.endswith("_low"):
                    possible_mappings[col] = "Low"
                elif lc.endswith("_close"):
                    possible_mappings[col] = "Close"
                elif lc.endswith("_adj close"):
                    possible_mappings[col] = "Adj Close"
                elif lc.endswith("_volume"):
                    possible_mappings[col] = "Volume"

            if possible_mappings:
                df.rename(columns=possible_mappings, inplace=True)

            # Create a business-day date range to ensure consistent daily rows
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')

            # Reindex and forward-fill missing rows
            df = df.reindex(date_range).ffill()

            df["title"] = self.title
            self.data = df.dropna(how="all")  # Drop rows that are all NaN (unlikely after fill)

        except Exception as e:
            raise RuntimeError(f"Error fetching price data for {self.symbol}: {e}")

        return self.data

    def calculate_performance(self, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Calculate daily return (%), daily absolute change, and cumulative return
        based on 'Close' prices. Appends columns to self.data and returns it.
        """
        if self.data.empty or "Close" not in self.data.columns:
            raise ValueError(f"No price data available for {self.symbol} to calculate performance.")

        df = self.data.copy()
        if start_date is not None and end_date is not None:
            df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
        initial_close = df["Close"].iloc[0]
        df["Daily Return (%)"] = df["Close"].pct_change() * 100
        df["Daily Absolute Change"] = df["Close"].diff()
        df["Performance (Abs)"] = df["Close"] - initial_close
        df["Performance (%)"] = (df["Close"] / initial_close - 1) * 100
        df['ticker'] = self.symbol
        df['title'] = self.title

        self.data = df
        return self.data

    def calculate_performance_by_date(self, start_date, end_date) -> pd.DataFrame:
        """
        Calculate daily return (%), daily absolute change, and cumulative return
        based on 'Close' prices. Appends columns to self.data and returns it.
        """
        if self.data.empty or "Close" not in self.data.columns:
            raise ValueError(f"No price data available for {self.symbol} to calculate performance.")

        df = self.data.copy()
        df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
        df["Daily Return (%)"] = df["Close"].pct_change() * 100
        df["Daily Absolute Change"] = df["Close"].diff()
        df['ticker'] = self.symbol
        df['title'] = self.title

        self.data = df
        return self.data

    def __repr__(self):
        return f"<Ticker {self.symbol} ({self.title})>"


# ----------------------------------------------
# 3) TICKER COLLECTION (Similar to TickerAnalysis)
# ----------------------------------------------
class TickerCollection:
    """
    Manages multiple Ticker objects: fetching data, computing performance, etc.
    """

    def __init__(self, symbols: List[str], start_date: date, end_date: date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.tickers_map: Dict[str, Ticker] = {}

        # Initialize Ticker objects (lazy-load data in separate methods)
        for symbol in self.symbols:
            self.tickers_map[symbol.upper()] = Ticker(symbol)

    def fetch_fundamental_data(self):
        """
        Fetch fundamental data for each ticker and store in each Ticker object.
        """
        for symbol, ticker in self.tickers_map.items():
            ticker.fetch_fundamental_data()

    def fetch_price_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch and store historical price data for all Ticker objects.
        Return a dict of symbol -> DataFrame.
        """
        all_data = {}
        for symbol, ticker in self.tickers_map.items():
            df = ticker.fetch_price_data(self.start_date, self.end_date)
            all_data[symbol] = df
        return all_data

    def calculate_all_ticker_performances(self, start_date=None, end_date=None) -> Dict[str, pd.DataFrame]:
        """
        Calculate performance metrics for each Ticker.
        Returns a dict keyed by symbol -> performance DataFrame.
        """
        performances = {}
        for symbol, ticker in self.tickers_map.items():
            df_perf = ticker.calculate_performance(start_date, end_date)
            performances[symbol] = df_perf.reset_index().rename(columns={'index': 'Date'})
        return performances

    def calculate_all_ticker_performances_by_date(self, start_date, end_date) -> Dict[str, pd.DataFrame]:
        """
        Calculate performance metrics for each Ticker.
        Returns a dict keyed by symbol -> performance DataFrame.
        """
        performances = {}
        for symbol, ticker in self.tickers_map.items():
            df_perf = ticker.calculate_performance_by_date(start_date, end_date)
            performances[symbol] = df_perf.reset_index().rename(columns={'index': 'Date'})
        return performances

    def get_ticker_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Retrieve the stored price data for a given symbol.
        """
        ticker = self.tickers_map.get(symbol.upper())
        if ticker:
            return ticker.data
        return None

    def __repr__(self):
        return f"<TickerCollection {list(self.tickers_map.keys())}>"


class Benchmark(Ticker):
    """
    A Benchmark is a specialized Ticker without transactional data,
    used for tracking indices or ETF performance.
    """

    def __init__(self, symbol: str):
        # Initialize the parent Ticker class
        super().__init__(symbol)

    def fetch_fundamental_data(self):
        """
        Override the fetch_fundamental_data method for benchmarks.
        Benchmarks may not have the same fundamental data as individual stocks.
        """
        try:
            stock = yf.Ticker(self.symbol)
            info = stock.info  # Deprecated in yfinance >= 0.2.4, but still works for now

            self.fundamentals = {
                "Quote Type": info.get("quoteType"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "Full Name": info.get("longName"),
            }

            if self.fundamentals.get("Full Name"):
                self.title = self.fundamentals["Full Name"]
            if self.fundamentals.get("Quote Type"):
                self.asset_type = self.fundamentals["Quote Type"]  # e.g., 'EQUITY'
            if self.fundamentals.get("Sector"):
                self.sector = self.fundamentals["Sector"]
            if self.fundamentals.get("Industry"):
                self.industry = self.fundamentals["Industry"]

        except Exception as e:
            self.fundamentals = {"Error": str(e)}

    def calculate_performance(self, start_date=None, end_date=None) -> pd.DataFrame:
        return super().calculate_performance(start_date, end_date)

    def __repr__(self):
        return f"<Benchmark {self.symbol} ({self.title})>"


class BenchmarkCollection(TickerCollection):
    """
    A collection of Benchmarks, reusing TickerCollection logic with minor adjustments.
    """

    def __init__(self, symbols: List[str], start_date: date, end_date: date):
        super().__init__(symbols, start_date, end_date)
        # Replace Ticker instances with Benchmark instances
        self.tickers_map = {symbol: Benchmark(symbol) for symbol in symbols}

    def compute_all_ticker_performances(self, start_date=None, end_date=None):
        return super().calculate_all_ticker_performances(start_date, end_date)

    def compute_all_ticker_performances_by_date(self, start_date, end_date):
        return super().calculate_all_ticker_performances_by_date(start_date, end_date)

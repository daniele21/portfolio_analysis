from datetime import date
from typing import Dict, List, Any

import pandas as pd
import yfinance as yf

from portfolio_analysis.scripts.data.ticker import Ticker
from portfolio_analysis.scripts.data.transaction import Transaction


class Portfolio:
    """
    A user's portfolio containing multiple transactions across various tickers.
    Uses Ticker data (via TickerCollection) to compute portfolio performance.
    """

    def __init__(self, name: str, transactions: pd.DataFrame, tickers_map: Dict[str, Ticker]):
        """
        :param name: Name of the portfolio, e.g., "My Investments"
        :param transactions: DataFrame with columns:
                             [Operation, Date, Valuta, Type, Ticker, Quantity]
        :param tickers_map:  Dictionary of symbol -> Ticker objects
        """
        self.name = name
        self.tickers_map = tickers_map
        self.transactions_df = transactions.copy()

        # Convert the DataFrame rows into Transaction objects (optional, if you prefer OOP style)
        self.transactions: List[Transaction] = []
        for _, row in self.transactions_df.iterrows():
            txn = Transaction(
                operation=row["Operation"],
                date_=row["Date"],
                valuta=row["Valuta"],
                ticker_symbol=row["Ticker"],
                quantity=row["Quantity"]
            )
            self.transactions.append(txn)

        # Will hold the main portfolio-level DataFrame and per-ticker performance
        self.portfolio_df: pd.DataFrame = pd.DataFrame()
        self.all_tickers_perf: Dict[str, pd.DataFrame] = {}

    def calculate_portfolio_performance_by_date(self, start_date: date, end_date: date):
        """
        Calculate the portfolio value over time within a specified date range,
        taking into account all transactions and the Ticker close prices.

        :param start_date: Start date for the performance calculation.
        :param end_date: End date for the performance calculation.

        Returns:
            portfolio (DataFrame): Overall portfolio performance with columns:
                [Date, Total Value, Cost Basis, Unrealized Gains, Realized Gains, Daily Performance (%), ...]
            all_tickers_perf (Dict[str, DataFrame]): Detailed Ticker-level performance
        """
        # --- 1) Group transactions by ticker for easier iteration ---
        transactions_by_ticker = {}
        for txn in self.transactions:
            # if pd.Timestamp(start_date) <= txn.date <= pd.Timestamp(end_date):
                transactions_by_ticker.setdefault(txn.ticker_symbol, []).append(txn)

        # Sort each ticker's transactions by date
        for ticker_symbol in transactions_by_ticker:
            transactions_by_ticker[ticker_symbol].sort(key=lambda tx: tx.date)

        # --- 2) Build a performance DataFrame for each ticker ---
        all_tickers_perf = {}
        for symbol, ticker in self.tickers_map.items():
            # If we don't have any historical data for this ticker, skip
            if ticker.data.empty:
                continue

            # Filter ticker data to the date range
            df_prices = ticker.data.copy()
            df_prices = df_prices.loc[(df_prices.index.date >= start_date) & (df_prices.index.date <= end_date)]

            if df_prices.empty:
                continue

            df_prices["Quantity Held"] = 0.0
            df_prices["Cost Basis"] = 0.0
            df_prices["Unrealized Gains"] = 0.0
            df_prices["Realized Gains"] = 0.0
            df_prices["Market Value"] = 0.0
            df_prices["Performance (%)"] = 0.0
            df_prices["Ticker"] = symbol
            df_prices["Title"] = ticker.title
            df_prices["AssetType"] = ticker.asset_type or "Unknown"
            df_prices["Sector"] = ticker.sector or "Unknown"
            df_prices["Industry"] = ticker.industry or "Unknown"

            # Go through transactions in chronological order
            ticker_txns = transactions_by_ticker.get(symbol, [])
            quantity_held = 0.0
            total_cost = 0.0
            realized_gains = 0.0
            tx_index = 0

            # For each date in this ticker's price data
            for dt in df_prices.index:
                # Process all transactions up to (and including) this date
                while tx_index < len(ticker_txns) and ticker_txns[tx_index].date <= dt:
                    txn = ticker_txns[tx_index]
                    price_at_tx = df_prices.loc[dt, "Close"]

                    if txn.operation == "buy":
                        quantity_held += txn.quantity
                        total_cost += txn.quantity * price_at_tx
                    elif txn.operation == "sell":
                        if quantity_held > 0:
                            avg_cost = total_cost / quantity_held
                            # Realized gain for the sold quantity
                            realized_gains += txn.quantity * (price_at_tx - avg_cost)
                            quantity_held -= txn.quantity
                            total_cost -= txn.quantity * avg_cost

                    tx_index += 1

                # Update the columns for today's row
                df_prices.loc[dt, "Quantity Held"] = quantity_held
                df_prices.loc[dt, "Cost Basis"] = total_cost
                df_prices.loc[dt, "Realized Gains"] = realized_gains

                if quantity_held > 0:
                    mv = quantity_held * df_prices.loc[dt, "Close"]
                    df_prices.loc[dt, "Market Value"] = mv
                    df_prices.loc[dt, "Unrealized Gains"] = mv - total_cost
                    # Basic performance as (MV - cost) / cost
                    if total_cost > 0:
                        df_prices.loc[dt, "Performance (%)"] = 100 * (mv - total_cost) / total_cost

            all_tickers_perf[symbol] = df_prices.reset_index().rename(columns={"index": "Date"})

        # --- 3) Aggregate into a single portfolio DataFrame ---
        # Collect all distinct dates
        all_dates = sorted(set().union(*(df["Date"] for df in all_tickers_perf.values())))
        all_dates = [dt for dt in all_dates if start_date <= dt.date() <= end_date]

        portfolio = pd.DataFrame(
            columns=["Total Value", "Cost Basis", "Unrealized Gains", "Realized Gains", "Daily Performance (%)"],
            index=all_dates
        )

        prev_total_value = 0.0
        for dt in all_dates:
            total_value = 0.0
            cost_basis = 0.0
            unreal_gains_sum = 0.0
            realized_gains_sum = 0.0

            # Sum across all tickers
            for symbol, df_perf in all_tickers_perf.items():
                row_today = df_perf.loc[df_perf["Date"] == dt]
                if not row_today.empty:
                    total_value += row_today["Market Value"].values[0]
                    cost_basis += row_today["Cost Basis"].values[0]
                    unreal_gains_sum += row_today["Unrealized Gains"].values[0]
                    realized_gains_sum += row_today["Realized Gains"].values[0]

            portfolio.loc[dt, "Total Value"] = total_value
            portfolio.loc[dt, "Cost Basis"] = cost_basis
            portfolio.loc[dt, "Unrealized Gains"] = unreal_gains_sum
            portfolio.loc[dt, "Realized Gains"] = realized_gains_sum

            # Daily performance % (compared to previous day)
            if prev_total_value != 0:
                daily_perf = ((total_value - prev_total_value) / prev_total_value) * 100
            else:
                daily_perf = 0.0
            portfolio.loc[dt, "Daily Performance (%)"] = daily_perf

            prev_total_value = total_value

        # Clean up index
        portfolio.reset_index(inplace=True)
        portfolio.rename(columns={"index": "Date"}, inplace=True)

        # Example additional performance metric
        portfolio["Performance (%)"] = pd.Series(0.0, index=portfolio.index, dtype="float64")

        # Perform the calculation, ensuring the column and operations align with float64
        portfolio.loc[portfolio["Cost Basis"] > 0, "Performance (%)"] = (
                portfolio.loc[portfolio["Cost Basis"] > 0, "Unrealized Gains"] /
                portfolio.loc[portfolio["Cost Basis"] > 0, "Cost Basis"] * 100
        )

        self.portfolio_df = portfolio
        self.all_tickers_perf = all_tickers_perf

        return portfolio, all_tickers_perf

    def calculate_portfolio_performance(self):
        """
        Calculate the portfolio value over time, taking into account
        all transactions and the Ticker close prices.

        Returns:
            portfolio (DataFrame): Overall portfolio performance with columns:
                [Date, Total Value, Cost Basis, Unrealized Gains, Realized Gains, Daily Performance (%), ...]
            all_tickers_perf (Dict[str, DataFrame]): Detailed Ticker-level performance
        """
        # --- 1) Group transactions by ticker for easier iteration ---
        transactions_by_ticker = {}
        for txn in self.transactions:
            transactions_by_ticker.setdefault(txn.ticker_symbol, []).append(txn)

        # Sort each ticker's transactions by date
        for ticker_symbol in transactions_by_ticker:
            transactions_by_ticker[ticker_symbol].sort(key=lambda tx: tx.date)

        # --- 2) Build a performance DataFrame for each ticker ---
        all_tickers_perf = {}
        for symbol, ticker in self.tickers_map.items():
            # If we don't have any historical data for this ticker, skip
            if ticker.data.empty:
                continue

            df_prices = ticker.data.copy()
            df_prices["Quantity Held"] = 0.0
            df_prices["Cost Basis"] = 0.0
            df_prices["Unrealized Gains"] = 0.0
            df_prices["Realized Gains"] = 0.0
            df_prices["Market Value"] = 0.0
            df_prices["Performance (%)"] = 0.0
            df_prices["Ticker"] = symbol
            df_prices["Title"] = ticker.title
            df_prices["AssetType"] = ticker.asset_type or "Unknown"
            df_prices["Sector"] = ticker.sector or "Unknown"
            df_prices["Industry"] = ticker.industry or "Unknown"

            # Go through transactions in chronological order
            ticker_txns = transactions_by_ticker.get(symbol, [])
            quantity_held = 0.0
            total_cost = 0.0
            realized_gains = 0.0
            tx_index = 0

            # For each date in this ticker's price data
            for dt in df_prices.index:
                # Process all transactions up to (and including) this date
                while tx_index < len(ticker_txns) and ticker_txns[tx_index].date <= dt:
                    txn = ticker_txns[tx_index]
                    price_at_tx = df_prices.loc[dt, "Close"]

                    if txn.operation == "buy":
                        quantity_held += txn.quantity
                        total_cost += txn.quantity * price_at_tx
                    elif txn.operation == "sell":
                        if quantity_held > 0:
                            avg_cost = total_cost / quantity_held
                            # Realized gain for the sold quantity
                            realized_gains += txn.quantity * (price_at_tx - avg_cost)
                            quantity_held -= txn.quantity
                            total_cost -= txn.quantity * avg_cost

                    tx_index += 1

                # Update the columns for today's row
                df_prices.loc[dt, "Quantity Held"] = quantity_held
                df_prices.loc[dt, "Cost Basis"] = total_cost
                df_prices.loc[dt, "Realized Gains"] = realized_gains

                if quantity_held > 0:
                    mv = quantity_held * df_prices.loc[dt, "Close"]
                    df_prices.loc[dt, "Market Value"] = mv
                    df_prices.loc[dt, "Unrealized Gains"] = mv - total_cost
                    # Basic performance as (MV - cost) / cost
                    if total_cost > 0:
                        df_prices.loc[dt, "Performance (%)"] = 100*(mv - total_cost) / total_cost

            all_tickers_perf[symbol] = df_prices.reset_index().rename(columns={"index": "Date"})

        # --- 3) Aggregate into a single portfolio DataFrame ---
        # Collect all distinct dates
        all_dates = sorted(set().union(*(df["Date"] for df in all_tickers_perf.values())))
        portfolio = pd.DataFrame(
            columns=["Total Value", "Cost Basis", "Unrealized Gains", "Realized Gains", "Daily Performance (%)"],
            index=all_dates
        )

        prev_total_value = 0.0
        for dt in all_dates:
            total_value = 0.0
            cost_basis = 0.0
            unreal_gains_sum = 0.0
            realized_gains_sum = 0.0

            # Sum across all tickers
            for symbol, df_perf in all_tickers_perf.items():
                row_today = df_perf.loc[df_perf["Date"] == dt]
                if not row_today.empty:
                    total_value += row_today["Market Value"].values[0]
                    cost_basis += row_today["Cost Basis"].values[0]
                    unreal_gains_sum += row_today["Unrealized Gains"].values[0]
                    realized_gains_sum += row_today["Realized Gains"].values[0]

            portfolio.loc[dt, "Total Value"] = total_value
            portfolio.loc[dt, "Cost Basis"] = cost_basis
            portfolio.loc[dt, "Unrealized Gains"] = unreal_gains_sum
            portfolio.loc[dt, "Realized Gains"] = realized_gains_sum

            # Daily performance % (compared to previous day)
            if prev_total_value != 0:
                daily_perf = ((total_value - prev_total_value) / prev_total_value) * 100
            else:
                daily_perf = 0.0
            portfolio.loc[dt, "Daily Performance (%)"] = daily_perf

            prev_total_value = total_value

        # Clean up index
        portfolio.reset_index(inplace=True)
        portfolio.rename(columns={"index": "Date"}, inplace=True)

        # Example additional performance metric
        portfolio["Performance (%)"] = 0.0
        portfolio.loc[portfolio["Cost Basis"] > 0, "Performance (%)"] = (
                portfolio["Unrealized Gains"] / portfolio["Cost Basis"] * 100
        )

        self.portfolio_df = portfolio
        self.all_tickers_perf = all_tickers_perf

        return portfolio, all_tickers_perf

    # -------------------------------------
    # 4A) HELPER METHODS FOR KPI COMPUTATION
    # -------------------------------------
    def compute_daily_returns(self) -> pd.Series:
        """
        Return the daily returns as a Series in decimal form, e.g. 0.01 for 1%.
        Computed from 'Daily Performance (%)' in the self.portfolio_df.
        """
        if self.portfolio_df.empty:
            raise ValueError("Portfolio data is empty. Run calculate_portfolio_performance() first.")

        # Convert daily performance % to decimal
        daily_returns = self.portfolio_df["Daily Performance (%)"] / 100.0
        return daily_returns

    def compute_abs_return(self) -> float:
        """
        Compute absolute return from the first day to the last.
        """
        if self.portfolio_df.empty:
            raise ValueError("Portfolio data is empty. Run calculate_portfolio_performance() first.")

        initial_value = self.portfolio_df["Total Value"].iloc[0]
        latest_value = self.portfolio_df["Total Value"].iloc[-1]
        if initial_value == 0:
            return 0.0
        abs_return = (latest_value - initial_value) / initial_value * 100
        return abs_return

    def compute_volatility(self, annualize: bool = True) -> float:
        """
        Calculate the standard deviation of daily returns. Annualize if desired.
        """
        daily_returns = self.compute_daily_returns()
        vol = daily_returns.std()
        if annualize:
            # ~252 trading days in a year
            vol *= (252 ** 0.5)
        return vol

    def compute_sharpe_ratio(self, risk_free_rate_annual: float = 0.01) -> float:
        """
        Compute the Sharpe Ratio based on daily returns and a given annual risk-free rate.
        """
        daily_returns = self.compute_daily_returns()
        # Convert the annual risk-free rate to a daily figure
        daily_rf = risk_free_rate_annual / 252.0
        excess_returns = daily_returns - daily_rf
        if excess_returns.std() == 0:
            return 0.0
        sharpe = (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)
        return sharpe

    def compute_asset_allocation(self) -> pd.DataFrame:
        """
        Returns a DataFrame showing final-day allocation by AssetType (or 'Unknown').
        """
        if not self.all_tickers_perf:
            raise ValueError("No ticker performance data. Run calculate_portfolio_performance() first.")

        # We'll look at the last row for each ticker
        allocations = []
        total_portfolio_value = self.portfolio_df["Total Value"].iloc[-1]
        for symbol, df_ in self.all_tickers_perf.items():
            if df_.empty:
                continue
            last_row = df_.iloc[-1]
            mv = last_row["Market Value"]
            asset_type = last_row["AssetType"]
            alloc_percent = (mv / total_portfolio_value * 100) if total_portfolio_value != 0 else 0
            allocations.append((symbol, asset_type, mv, alloc_percent))

        df_alloc = pd.DataFrame(allocations, columns=["Ticker", "AssetType", "MarketValue", "Allocation(%)"])
        # If we want to group by asset type:
        # df_alloc = df_alloc.groupby("AssetType").sum(numeric_only=True)
        return df_alloc

    def compute_sector_industry_breakdown(self) -> pd.DataFrame:
        """
        Returns a DataFrame showing final-day allocation by Sector and Industry.
        """
        if not self.all_tickers_perf:
            raise ValueError("No ticker performance data. Run calculate_portfolio_performance() first.")

        breakdowns = []
        total_portfolio_value = self.portfolio_df["Total Value"].iloc[-1]
        for symbol, df_ in self.all_tickers_perf.items():
            if df_.empty:
                continue
            last_row = df_.iloc[-1]
            mv = last_row["Market Value"]
            sector = last_row["Sector"]
            industry = last_row["Industry"]
            alloc_percent = (mv / total_portfolio_value * 100) if total_portfolio_value != 0 else 0
            breakdowns.append((symbol, sector, industry, mv, alloc_percent))

        df_sector = pd.DataFrame(breakdowns, columns=["Ticker", "Sector", "Industry", "MarketValue", "Allocation(%)"])
        return df_sector

    def compute_concentration_risk(self) -> pd.DataFrame:
        """
        Returns a DataFrame of tickers sorted by their final-day % of the total portfolio.
        Helps identify large positions that might be a concentration risk.
        """
        df_alloc = self.compute_asset_allocation()
        df_alloc.sort_values(by="Allocation(%)", ascending=False, inplace=True)
        return df_alloc

    def compute_contribution_to_return(self) -> pd.DataFrame:
        """
        Approximate each ticker's contribution to portfolio returns over time.
        - We use a 'weighted daily return' approach.
        """
        if not self.all_tickers_perf:
            raise ValueError("No ticker performance data. Run calculate_portfolio_performance() first.")

        # 1. We need the daily total portfolio value to compute weights
        portfolio_df = self.portfolio_df.set_index("Date")

        # 2. For each ticker, compute daily return in decimal form
        contributions = []
        for symbol, df_ in self.all_tickers_perf.items():
            df_temp = df_.copy()
            df_temp.set_index("Date", inplace=True)

            # Ticker's daily raw return in decimal
            # We'll approximate from day to day based on "Market Value"
            # daily_ticker_return = df_temp["Market Value"].pct_change() # if you want MV-based
            # But let's keep it consistent with "Performance" (which is fraction-based)
            # df_temp["Daily Ticker Return"] = df_temp["Performance"]  # This is fraction (not %)
            # or we can recast from % if needed

            df_temp["Daily Ticker Return"] = df_temp["Performance (%)"]  # Performance is fraction, e.g. 0.05 = 5%
            df_temp["WeightInPortfolio"] = (
                    df_temp["Market Value"] / portfolio_df["Total Value"]
            )

            # Ticker's contribution on each day = daily_return * weight
            df_temp["Daily Contribution"] = df_temp["Daily Ticker Return"] * df_temp["WeightInPortfolio"]

            # Summation over the entire period
            total_contribution = df_temp["Daily Contribution"].sum()
            # Convert to a "percentage point" or keep as fraction
            # e.g., 0.10 means 10% total contribution
            contributions.append((symbol, total_contribution))

        df_contrib = pd.DataFrame(contributions, columns=["Ticker", "Contribution"])
        df_contrib.sort_values("Contribution", ascending=False, inplace=True)
        return df_contrib

    def compute_benchmark_comparison(self, benchmark_symbol: str) -> pd.DataFrame:
        """
        Fetch benchmark data and compare the daily returns side-by-side with the portfolio.
        Return a DataFrame with columns: [Date, Portfolio Daily %, Benchmark Daily %].
        """
        # 1. Fetch the benchmark as a Ticker
        benchmark_ticker = Ticker(benchmark_symbol)
        start_date = self.portfolio_df["Date"].min()
        end_date = self.portfolio_df["Date"].max()
        benchmark_ticker.fetch_price_data(start_date, end_date)
        benchmark_ticker.calculate_performance()

        # 2. Build a daily returns DataFrame
        bench_df = benchmark_ticker.data.copy()
        bench_df["Daily Bench Return (%)"] = bench_df["Close"].pct_change() * 100
        bench_df.reset_index(inplace=True)
        bench_df.rename(columns={"index": "Date"}, inplace=True)

        # 3. Merge with portfolio daily performance
        portfolio_df = self.portfolio_df[["Date", "Daily Performance (%)"]].copy()
        df_compare = pd.merge(portfolio_df, bench_df[["Date", "Daily Bench Return (%)"]], on="Date", how="outer")
        df_compare.fillna(0, inplace=True)
        return df_compare

    def __repr__(self):
        return f"<Portfolio name={self.name}, Transactions={len(self.transactions)}>"


def calculate_kpis(portfolio_df: pd.DataFrame,
                   allocation_df: pd.DataFrame,
                   all_tickers_perf: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Calculate key performance indicators for the portfolio.
    :param portfolio_df: The aggregated portfolio performance DataFrame.
    :param allocation_df: A DataFrame with columns like
                          [Ticker, Quantity Held, Market Value, Unrealized Gains, Realized Gains].
    :param all_tickers_perf: Dict keyed by ticker, each DataFrame has 'Performance' or 'Performance (%)'.
    """
    total_value = portfolio_df["Total Value"].iloc[-1]
    performance_pct = portfolio_df["Performance (%)"].iloc[-1]
    unrealized_gains = allocation_df["Unrealized Gains"].sum()
    realized_gains = allocation_df["Realized Gains"].sum()

    # Daily change in portfolio value
    if len(portfolio_df) > 1:
        prev_value = portfolio_df["Total Value"].iloc[-2]
        daily_change = ((total_value - prev_value) / prev_value) * 100 if prev_value else 0
    else:
        daily_change = 0

    # Identify best/worst ticker (by unrealized gains)
    best_ticker_row = allocation_df.loc[allocation_df["Unrealized Gains"].idxmax()]
    worst_ticker_row = allocation_df.loc[allocation_df["Unrealized Gains"].idxmin()]

    best_ticker_symbol = best_ticker_row["Ticker"]
    worst_ticker_symbol = worst_ticker_row["Ticker"]

    # If your all_tickers_perf includes a "Performance" or "Performance (%)" column,
    # you can also compare that here. For example:
    best_perf_df = all_tickers_perf[best_ticker_symbol]
    best_perf_col = "Performance" if "Performance" in best_perf_df.columns else "Performance (%)"
    best_ticker_perf = best_perf_df[best_perf_col].iloc[-1]  # or best_perf_df["Performance (%)"].iloc[-1]

    worst_perf_df = all_tickers_perf[worst_ticker_symbol]
    worst_perf_col = "Performance" if "Performance" in worst_perf_df.columns else "Performance (%)"
    worst_ticker_perf = worst_perf_df[worst_perf_col].iloc[-1]

    best_ticker = {
        "ticker": best_ticker_symbol,
        "unrealized_gains": best_ticker_row["Unrealized Gains"],
        "performance": best_ticker_perf
    }
    worst_ticker = {
        "ticker": worst_ticker_symbol,
        "unrealized_gains": worst_ticker_row["Unrealized Gains"],
        "performance": worst_ticker_perf
    }

    return {
        "value": total_value,
        "unrealized_gains": unrealized_gains,
        "realized_gains": realized_gains,
        "performance": performance_pct,
        "daily_change": daily_change,
        "best_ticker": best_ticker,
        "worst_ticker": worst_ticker,
    }
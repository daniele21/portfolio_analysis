import yfinance as yf
import pandas as pd


class TickerAnalysis:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.fetch_fundamental_data()

    def fetch_data(self):
        """
        Fetch historical price data (Open, High, Low, Close, Adj Close, Volume)
        for all tickers between self.start_date and self.end_date.
        Store results in self.data, a dict keyed by the full name of the ticker
        (if available), where each value is a DataFrame with a DateTimeIndex.
        """
        try:
            # Fetch fundamental data to get full names
            self.fetch_fundamental_data()

            # Download data in a single call
            raw_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                group_by='ticker'
            )

            if raw_data.empty:
                raise ValueError("No data was fetched for the given tickers and date range.")

            # If multiple tickers, yfinance returns a MultiIndex DataFrame.
            # If only one ticker, it returns a single-index DataFrame.
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Multi-ticker scenario
                for ticker in self.tickers:
                    if ticker in raw_data.columns.get_level_values(0):
                        df_ticker = raw_data[ticker].dropna(how='all')
                        if not df_ticker.empty:
                            # Use full name as key if available
                            full_name = self.fundamentals.get(ticker, {}).get("Full Name", ticker)
                            df_ticker.insert(df_ticker.shape[1], 'title', full_name)
                            self.data[ticker] = df_ticker
            else:
                # Single-ticker scenario
                raw_data.columns = [col.title() for col in raw_data.columns]
                full_name = self.fundamentals.get(self.tickers[0], {}).get("Full Name", self.tickers[0])
                self.data[full_name] = raw_data.dropna(how='all')

            if not self.data:
                raise ValueError("No valid data was found for the tickers.")

            # Optional: unify date indexes across all tickers
            all_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
            for ticker, df in self.data.items():
                # Reindex to have a consistent DateTimeIndex for all
                # and forward-fill missing data if any
                self.data[ticker] = df.reindex(all_dates).ffill()

            return self.data

        except Exception as e:
            raise RuntimeError(f"An error occurred while fetching data: {e}")

    def calculate_ticker_performance(self):
        """
        For each ticker in self.data, compute:
        - Daily Return (%)
        - Daily Absolute Change
        - Cumulative Return (%)

        Returns a dict keyed by ticker, each containing a DataFrame
        with columns:
            [Close, Daily Return (%), Daily Absolute Change, Cumulative Return (%)]
        """
        performance = {}
        for ticker, df in self.data.items():
            # Work on a copy so we don't alter the original
            df_perf = df.copy()

            # Calculate daily return and absolute change
            df_perf['Daily Return (%)'] = df_perf['Close'].pct_change() * 100
            df_perf['Daily Absolute Change'] = df_perf['Close'].diff()

            # Calculate cumulative returns
            df_perf['Cumulative Return (%)'] = (1 + df_perf['Close'].pct_change()).cumprod() - 1

            performance[ticker] = df_perf[[
                'Close',
                'Daily Return (%)',
                'Daily Absolute Change',
                'Cumulative Return (%)'
            ]]

        return performance

    def fetch_fundamental_data(self):
        """
        Fetch fundamental data (like financials, balance sheet, income statement,
        cashflow, market cap, PE ratio, sector, etc.) for all tickers.
        Store results in self.fundamentals, a dict keyed by ticker.
        """
        self.fundamentals = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)

                # Extract relevant data
                self.fundamentals[ticker] = {
                    # General Information
                    "Quote Type": stock.info.get("quoteType"),  # Asset type: ETF, Equity, Bond, etc.
                    "Sector": stock.info.get("sector"),
                    "Industry": stock.info.get("industry"),
                    "Country": stock.info.get("country"),
                    "Full Name": stock.info.get("longName"),
                    "Symbol": stock.info.get("symbol"),
                    "Exchange": stock.info.get("exchange"),
                    "Currency": stock.info.get("currency"),
                    # Financial Metrics
                    # "Market Cap": stock.info.get("marketCap"),
                    # "PE Ratio (Trailing)": stock.info.get("trailingPE"),
                    # "PE Ratio (Forward)": stock.info.get("forwardPE"),
                    # "EPS (Trailing)": stock.info.get("trailingEps"),
                    # "Dividend Yield": stock.info.get("dividendYield"),
                    # "Beta": stock.info.get("beta"),
                    # "Revenue (TTM)": stock.info.get("totalRevenue"),
                    # "Net Income (TTM)": stock.income_stmt.loc["Net Income"].sum()
                    # if "Net Income" in stock.income_stmt.index else None,
                    # "Debt to Equity": stock.info.get("debtToEquity"),
                    # "Profit Margins": stock.info.get("profitMargins"),
                    # "Free Cash Flow": stock.info.get("freeCashflow"),
                    # # Technical Information
                    # "52 Week High": stock.info.get("fiftyTwoWeekHigh"),
                    # "52 Week Low": stock.info.get("fiftyTwoWeekLow"),
                    # "200 Day Average": stock.info.get("twoHundredDayAverage"),
                    # "50 Day Average": stock.info.get("fiftyDayAverage"),
                    # # Financial Statements
                    # "Income Statement": stock.income_stmt,
                    # "Balance Sheet": stock.balance_sheet,
                    # "Cash Flow Statement": stock.cashflow,
                    # # Quarterly Financials
                    # "Quarterly Income Statement": stock.quarterly_income_stmt,
                    # "Quarterly Balance Sheet": stock.quarterly_balance_sheet,
                    # "Quarterly Cash Flow Statement": stock.quarterly_cashflow,
                    # # ESG and Sustainability
                    # "Sustainability": stock.sustainability,
                    # # Recommendations
                    # "Recommendations": stock.recommendations,
                    # # Holders
                    # "Major Holders": stock.major_holders,
                    # "Institutional Holders": stock.institutional_holders,
                    # # Dividends and Splits
                    # "Dividends": stock.dividends,
                    # "Splits": stock.splits,
                    # # Options
                    # "Options Expirations": stock.options
                }


            except Exception as e:
                self.fundamentals[ticker] = {"Error": str(e)}

        # return self.fundamentals

class PortfolioAnalysis:
    def __init__(self, transactions, ticker_data):
        """
        transactions: A DataFrame with columns:
            [Operation, Date, Valuta, Type, Ticker, Quantity, Importo]
        ticker_data: A dict of DataFrames keyed by ticker, each containing
                     at least a 'Close' column.
        """
        self.transactions = transactions
        self.ticker_data = ticker_data

    def calculate_portfolio_performance(self):
        """
        Calculate the portfolio value (and daily performance) over time using the ticker_data and transaction history.

        Returns:
            portfolio (DataFrame): Overall portfolio performance with columns:
                [Total Value, Cost Basis, Unrealized Gains, Realized Gains, Daily Performance (%)].
            consolidated_ticker_perf (DataFrame): Consolidated performance of each ticker with columns:
                [Ticker, Date, Quantity Held, Cost Basis, Realized Gains, Unrealized Gains, Market Value, Performance].
        """
        # --- Preprocess transactions: group by ticker for efficiency ---
        transactions_by_ticker = {}
        for ticker in self.transactions['Ticker'].unique():
            txns = self.transactions[self.transactions['Ticker'] == ticker].sort_values('Date')
            transactions_by_ticker[ticker] = txns

        # --- STEP 1: Build a performance DataFrame for each ticker ---
        performance = {}
        for ticker, df_prices in self.ticker_data.items():
            df_perf = df_prices.copy()
            title = df_prices['title'].iloc[-1]
            df_perf['Quantity Held'] = 0
            df_perf['Cost Basis'] = 0.0
            df_perf['Unrealized Gains'] = 0.0
            df_perf['Realized Gains'] = 0.0
            df_perf['Market Value'] = 0.0
            df_perf['Performance'] = 0.0

            ticker_txns = transactions_by_ticker.get(ticker, pd.DataFrame())
            quantity_held = 0
            total_cost = 0.0
            realized_gains = 0.0
            txns_idx = 0

            for date in df_perf.index:
                while txns_idx < len(ticker_txns) and ticker_txns.iloc[txns_idx]['Date'] <= date:
                    txn = ticker_txns.iloc[txns_idx]
                    op = txn['Operation'].lower()
                    qty = txn['Quantity']
                    price = df_perf.loc[date, 'Close']

                    if op == 'buy':
                        quantity_held += qty
                        total_cost += qty * price
                    elif op == 'sell':
                        if quantity_held > 0:
                            avg_cost = total_cost / quantity_held
                            realized_gains += qty * (price - avg_cost)
                            quantity_held -= qty
                            total_cost -= qty * avg_cost
                    txns_idx += 1

                df_perf.loc[date, 'Quantity Held'] = quantity_held
                df_perf.loc[date, 'Cost Basis'] = total_cost
                df_perf.loc[date, 'Realized Gains'] = realized_gains
                if quantity_held > 0:
                    market_value = quantity_held * df_perf.loc[date, 'Close']
                    df_perf.loc[date, 'Market Value'] = market_value
                    df_perf.loc[date, 'Unrealized Gains'] = market_value - total_cost
                    df_perf.loc[date, 'Performance'] = (market_value - total_cost) / total_cost if total_cost > 0 else 0

            df_perf['Ticker'] = ticker
            df_perf['Title'] = title
            performance[ticker] = df_perf.reset_index().rename(columns={'index': 'Date'})

        # --- STEP 2: Aggregate into a single portfolio DataFrame ---
        all_dates = sorted(set().union(*(performance[t]['Date'] for t in performance)))
        portfolio = pd.DataFrame(index=all_dates,
                                 columns=['Total Value', 'Cost Basis', 'Unrealized Gains', 'Realized Gains',
                                          'Daily Performance (%)'])
        portfolio[['Total Value', 'Realized Gains', 'Unrealized Gains', 'Performance']] = 0.0

        prev_total_value = None

        for date in all_dates:
            total_value = 0.0
            realized_gains_sum = 0.0
            unrealized_gains_sum = 0.0
            cost_basis = 0.0

            for ticker, perf_df in performance.items():
                if date in perf_df['Date'].values:
                    row = perf_df.loc[perf_df['Date'] == date].iloc[0]
                    total_value += row['Quantity Held'] * row['Close']
                    realized_gains_sum += row['Realized Gains']
                    unrealized_gains_sum += row['Unrealized Gains']
                    cost_basis += row['Cost Basis']

            portfolio.loc[date, 'Total Value'] = total_value
            portfolio.loc[date, 'Cost Basis'] = cost_basis
            portfolio.loc[date, 'Realized Gains'] = realized_gains_sum
            portfolio.loc[date, 'Unrealized Gains'] = unrealized_gains_sum

            if prev_total_value is not None and prev_total_value != 0:
                portfolio.loc[date, 'Daily Performance (%)'] = ((total_value - prev_total_value) / prev_total_value) * 100
            else:
                portfolio.loc[date, 'Daily Performance (%)'] = 0.0

            prev_total_value = total_value

        # Reset the index for portfolio DataFrame
        portfolio.reset_index(inplace=True)
        portfolio.rename(columns={'index': 'Date'}, inplace=True)
        portfolio['Performance'] = 100*portfolio['Unrealized Gains'] / portfolio['Cost Basis']

        return portfolio, performance



def calculate_kpis(portfolio_performance, portfolio_ticker_performance, allocation_df):
    """Calculate key performance indicators for the portfolio."""
    # Current portfolio total value
    total_value = portfolio_performance['Total Value'].iloc[-1]
    performance = portfolio_performance['Performance'].iloc[-1]

    # Total unrealized and realized gains
    unrealized_gains = allocation_df['Unrealized Gains'].sum()
    realized_gains = allocation_df['Realized Gains'].sum()

    # Daily change in portfolio value
    if len(portfolio_performance) > 1:
        previous_value = portfolio_performance['Total Value'].iloc[-2]
        daily_change = ((total_value - previous_value) / previous_value) * 100
    else:
        daily_change = 0

    # Best and worst performing tickers
    best_ticker = allocation_df.loc[allocation_df['Unrealized Gains'].idxmax()]
    worst_ticker = allocation_df.loc[allocation_df['Unrealized Gains'].idxmin()]

    best_perf = 100*portfolio_ticker_performance[best_ticker['Ticker']]['Performance'].iloc[-1]
    worst_perf = 100*portfolio_ticker_performance[worst_ticker['Ticker']]['Performance'].iloc[-1]

    best_ticker = {'ticker': best_ticker['Ticker'],
                   'unrealized_gains': best_ticker['Unrealized Gains'],
                   'performance': best_perf}
    worst_ticker = {'ticker': worst_ticker['Ticker'],
                    'unrealized_gains': worst_ticker['Unrealized Gains'],
                   'performance': worst_perf}

    portfolio_kpis = {'value': total_value,
                      'unrealized_gains': unrealized_gains,
                      'realized_gains': realized_gains,
                      'performance': performance,
                      'daily_change': daily_change,
                      'best_ticker': best_ticker,
                      'worst_ticker': worst_ticker}

    return portfolio_kpis
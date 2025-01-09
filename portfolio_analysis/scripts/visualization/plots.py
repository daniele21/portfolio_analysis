import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import qualitative as qc


# --------------------------------------------------------------------
# EXISTING / REFACTORED PLOTS
# --------------------------------------------------------------------

def create_transaction_annotated_line_chart(portfolio_df, transactions, name_col="Performance"):
    """
    Create a line chart of portfolio performance (e.g. 'Performance(%)') with annotations
    for buy/sell events, ensuring markers on the same date are slightly offset.

    :param portfolio_df: DataFrame with at least ['Date', name_col] columns.
    :param transactions: DataFrame with at least ['Operation', 'Date', 'Ticker'] columns.
    :param name_col: Column name in portfolio_df to plot (default: "Performance").
                     Could be "Daily Performance (%)", or "Performance (%)", etc.
    """
    fig = go.Figure()

    # 1) Line chart for the chosen performance metric
    fig.add_trace(go.Scatter(
        x=portfolio_df['Date'],
        y=portfolio_df[name_col],
        mode='lines',
        name=f'Portfolio {name_col}',
        line=dict(color='blue', width=2)
    ))

    # 2) Add buy/sell markers with slight vertical offsets
    marker_offsets = {}
    first_buy = True
    first_sell = True

    for _, txn in transactions.iterrows():
        event = txn['Operation'].lower()
        txn_date = txn['Date']
        performance_slice = portfolio_df.loc[portfolio_df['Date'] == txn_date, name_col]

        if not performance_slice.empty:
            performance_value = performance_slice.values[0]

            # Adjust y-value if the date already has markers
            if txn_date in marker_offsets:
                marker_offsets[txn_date] += 5
            else:
                marker_offsets[txn_date] = 0

            adjusted_val = performance_value + marker_offsets[txn_date]

            showlegend = False
            if event == 'buy' and first_buy:
                showlegend = True
                first_buy = False
            elif event == 'sell' and first_sell:
                showlegend = True
                first_sell = False

            fig.add_trace(go.Scatter(
                x=[txn_date],
                y=[adjusted_val],
                mode='markers+text',
                name="Buy" if event == "buy" else "Sell",
                marker=dict(
                    size=10,
                    color='green' if event == 'buy' else 'red',
                    symbol='triangle-up' if event == 'buy' else 'triangle-down'
                ),
                text=f"{event.capitalize()} {txn['Ticker']}",
                textposition="top center",
                showlegend=showlegend
            ))

    fig.update_layout(
        title=f"Portfolio {name_col} Over Time with Transactions",
        xaxis_title="Date",
        yaxis_title=name_col,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_close_price_with_transactions(ticker, ticker_performance_df, transactions):
    """
    Plot the close price of a ticker with grouped Buy/Sell transaction markers under a single legend entry.

    :param ticker: Ticker symbol (str)
    :param ticker_performance_df: DataFrame with at least ['Date', 'Close'] columns for this ticker
    :param transactions: DataFrame with at least ['Ticker', 'Operation', 'Date'] columns
    """
    if ticker_performance_df.empty:
        raise ValueError(f"No performance data found for ticker {ticker}")

    df = ticker_performance_df.copy()

    # Filter relevant transactions
    ticker_txns = transactions[transactions['Ticker'] == ticker]

    fig = go.Figure()

    # Plot close price
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))

    # Add buy/sell markers
    first_buy = True
    first_sell = True
    for _, txn in ticker_txns.iterrows():
        event = txn['Operation'].lower()
        txn_date = txn['Date']
        # Check if we have data for that date
        row_slice = df.loc[df['Date'] == txn_date, 'Close']
        if not row_slice.empty:
            price = row_slice.iloc[0]

            if pd.notna(price):
                if event == 'buy' and first_buy:
                    legend_name = 'Buy'
                    first_buy = False
                elif event == 'sell' and first_sell:
                    legend_name = 'Sell'
                    first_sell = False
                else:
                    legend_name = None

                fig.add_trace(go.Scatter(
                    x=[txn_date],
                    y=[price],
                    mode='markers+text',
                    name=legend_name,
                    marker=dict(
                        size=10,
                        color='green' if event == 'buy' else 'red',
                        symbol='triangle-up' if event == 'buy' else 'triangle-down'
                    ),
                    text=f"{event.capitalize()}",
                    textposition="top center",
                    showlegend=(legend_name is not None)
                ))

    fig.update_layout(
        title=f"Close Price and Transactions for {ticker}",
        xaxis_title="Date",
        yaxis_title="Close Price",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_unrealized_gains(ticker, ticker_performance_df, transactions):
    """
    Plot only the cumulative percentage of unrealized gains for a specific ticker,
    with Buy/Sell transaction markers aligned with that line.
    """
    if ticker_performance_df.empty:
        raise ValueError(f"No performance data found for ticker {ticker}")

    df = ticker_performance_df.copy()
    df = df[df['Quantity Held'] > 0]

    if df.empty:
        raise ValueError(f"No data for {ticker} with a non-zero Quantity Held.")

    # Calculate cumulative percentage of unrealized gains
    df['Cumulative Unrealized Gain (%)'] = (df['Unrealized Gains'] / df['Cost Basis']) * 100

    # Filter transactions for the selected ticker
    ticker_txns = transactions[transactions['Ticker'] == ticker]

    fig = go.Figure()

    # Plot cumulative percentage of unrealized gains
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Cumulative Unrealized Gain (%)'],
        mode='lines',
        name='Unrealized Gain (%)',
        line=dict(color='blue', width=2)
    ))

    # Add buy/sell markers
    first_buy = True
    first_sell = True
    for _, txn in ticker_txns.iterrows():
        event = txn['Operation'].lower()
        txn_date = txn['Date']
        row_slice = df.loc[df['Date'] == txn_date, 'Cumulative Unrealized Gain (%)']
        if not row_slice.empty:
            perf_val = row_slice.iloc[0]

            if pd.notna(perf_val):
                if event == 'buy' and first_buy:
                    legend_name = 'Buy'
                    first_buy = False
                elif event == 'sell' and first_sell:
                    legend_name = 'Sell'
                    first_sell = False
                else:
                    legend_name = None

                marker_symbol = 'triangle-up' if event == 'buy' else 'triangle-down'
                marker_color = 'green' if event == 'buy' else 'red'

                fig.add_trace(go.Scatter(
                    x=[txn_date],
                    y=[perf_val],
                    mode='markers+text',
                    name=legend_name,
                    marker=dict(
                        size=12,
                        color=marker_color,
                        symbol=marker_symbol,
                    ),
                    text=f"{event.capitalize()}",
                    textposition="top center",
                    showlegend=(legend_name is not None)
                ))

    fig.update_layout(
        title=f"Cumulative Unrealized Gain (%) for {ticker}",
        xaxis_title="Date",
        yaxis_title="Cumulative Unrealized Gain (%)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_pie_chart(allocation_df):
    """
    Plot the current portfolio allocation (by Ticker).
    Assumes allocation_df has columns ['Ticker', 'Market Value'].
    """
    fig = px.pie(allocation_df, names='Ticker', values='Market Value', title='Current Portfolio Allocation')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template='plotly_white')
    return fig


def create_bar_chart(allocation_df, value_col='Unrealized Gains', title='Value by Ticker'):
    """
    Plot a simple bar chart for the given value_col against Ticker.
    By default, it shows Unrealized Gains by Ticker.
    """
    fig = go.Figure(data=[
        go.Bar(name='Value', x=allocation_df['Ticker'], y=allocation_df[value_col], marker_color='blue')
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Amount",
        barmode='group',
        template='plotly_white'
    )
    return fig


# --------------------------------------------------------------------
# NEW / ADDITIONAL PLOTS FOR EXTRA KPI VISUALIZATIONS
# --------------------------------------------------------------------

def plot_portfolio_value_over_time(portfolio_df):
    """
    Line chart of the portfolio's Total Value over time.
    Expects portfolio_df with columns ['Date', 'Total Value'].
    """
    fig = px.line(portfolio_df, x='Date', y='Total Value', title="Total Portfolio Value Over Time")
    fig.update_layout(template='plotly_white', hovermode='x unified')
    return fig


def plot_daily_returns(portfolio_df, col='Daily Performance (%)', rolling_window=None):
    """
    Plots a bar (or line) chart of the daily returns.
    :param portfolio_df: DataFrame with at least ['Date', col] (in %)
    :param col: The column representing daily returns in percent, e.g. "Daily Performance (%)".
    :param rolling_window: Optionally compute a rolling average (e.g. 7 for 7-day).
    """
    df = portfolio_df.copy()
    df['Daily Returns'] = df[col]  # already in % or decimal
    if rolling_window:
        # e.g. compute rolling average
        df['RollingAvg'] = df['Daily Returns'].rolling(rolling_window).mean()

    fig = go.Figure()
    # Plot as bar for daily returns
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Daily Returns'],
        name='Daily Returns'
    ))

    # Optionally add a line for rolling average
    if rolling_window:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['RollingAvg'],
            mode='lines',
            line=dict(color='red', width=2),
            name=f'{rolling_window}-day Avg'
        ))

    fig.update_layout(
        title="Daily Returns",
        xaxis_title="Date",
        yaxis_title="Daily Returns (%)",
        template='plotly_white',
        hovermode='x unified'
    )
    return fig


def plot_bar_realized_vs_unrealized(allocation_df):
    """
    Plot a grouped bar chart showing Realized Gains vs. Unrealized Gains by Ticker.
    Assumes allocation_df has columns: ['Ticker', 'Unrealized Gains', 'Realized Gains'].
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=allocation_df['Ticker'],
        y=allocation_df['Realized Gains'],
        name='Realized Gains',
        marker_color='orange'
    ))
    fig.add_trace(go.Bar(
        x=allocation_df['Ticker'],
        y=allocation_df['Unrealized Gains'],
        name='Unrealized Gains',
        marker_color='blue'
    ))

    fig.update_layout(
        title="Realized vs. Unrealized Gains by Ticker",
        xaxis_title="Ticker",
        yaxis_title="Amount",
        barmode='group',
        template='plotly_white',
        hovermode='x unified'
    )
    return fig


def plot_portfolio_vs_benchmark(portfolio_df, benchmark_df,
                                portfolio_col='Daily Performance (%)',
                                benchmark_col='Daily Bench Return (%)'):
    """
    Plot a line chart comparing the portfolio's daily returns vs. a benchmark's daily returns.
    Both columns should be in percent form.

    :param portfolio_df: DataFrame with ['Date', portfolio_col]
    :param benchmark_df: DataFrame with ['Date', benchmark_col]
    :param portfolio_col: column name in portfolio_df for daily returns
    :param benchmark_col: column name in benchmark_df for daily returns
    """
    # Merge them on Date
    df_compare = pd.merge(portfolio_df[['Date', portfolio_col]],
                          benchmark_df[['Date', benchmark_col]],
                          on='Date', how='outer').fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_compare['Date'],
        y=df_compare[portfolio_col],
        mode='lines',
        name='Portfolio',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df_compare['Date'],
        y=df_compare[benchmark_col],
        mode='lines',
        name='Benchmark',
        line=dict(color='red')
    ))

    fig.update_layout(
        title="Portfolio vs. Benchmark Daily Returns (%)",
        xaxis_title="Date",
        yaxis_title="Daily Returns (%)",
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_cumulative_returns(
    portfolio_df,
    benchmarks=None,
    portfolio_label="Portfolio",
    benchmark_labels=None
):
    """
    Create a line chart showing 'Cumulative Return (%)' over time
    for the portfolio and (optionally) multiple benchmarks.

    :param portfolio_df: DataFrame with columns ["Date", "Cumulative Return (%)"].
    :param benchmarks: A list of DataFrames, each with ["Date", "Cumulative Return (%)"]. (optional)
    :param portfolio_label: Legend label for the portfolio line.
    :param benchmark_labels: A list of legend labels for the benchmark lines.
                             If None, defaults to ["Benchmark 1", "Benchmark 2", ...].
    """
    # 1) Initialize the figure
    fig = go.Figure()

    # 2) Plot the portfolio line
    if "Performance (%)" not in portfolio_df.columns:
        raise ValueError("portfolio_df must have a 'Performance (%)' column.")

    fig.add_trace(go.Scatter(
        x=portfolio_df["Date"],
        y=portfolio_df["Performance (%)"],
        mode='lines',
        name=portfolio_label,
        line=dict(color='blue', width=2)
    ))

    # 3) Handle multiple benchmarks
    if benchmarks is not None:
        # If no labels provided, create default ones
        if benchmark_labels is None:
            benchmark_labels = [f"Benchmark {i+1}" for i in range(len(benchmarks))]

        if len(benchmarks) != len(benchmark_labels):
            raise ValueError("Length of benchmarks and benchmark_labels must match.")

        # Use a color cycle from Plotly's qualitative palette
        color_cycle = qc.Plotly  # or qc.D3, qc.Set1, etc.

        for i, bench_df in enumerate(benchmarks):
            label = benchmark_labels[i]
            color = color_cycle[i % len(color_cycle)]

            if "Performance (%)" not in bench_df.columns:
                raise ValueError(f"Benchmark {i} missing 'Performance (%)' column.")

            # Merge with portfolio dates to align and ffill missing values
            merged = portfolio_df[["Date"]].merge(
                bench_df[["Date", "Performance (%)"]],
                on="Date", how="outer"
            ).sort_values("Date")

            merged.fillna(method='ffill', inplace=True)  # forward-fill
            merged.fillna(0, inplace=True)               # fallback if forward-fill can't apply

            fig.add_trace(go.Scatter(
                x=merged["Date"],
                y=merged["Performance (%)"],
                mode='lines',
                name=label,
                line=dict(color=color, width=2)
            ))

    # 4) Final layout
    fig.update_layout(
        title="Performance (%) Over Time",
        xaxis_title="Date",
        yaxis_title="Performance (%)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def plot_portfolio_and_tickers_performance(portfolio_df, all_tickers_perf,
                                           portfolio_col='Performance',
                                           ticker_col='Performance',
                                           portfolio_label='Portfolio'):
    """
    Plot a single figure comparing the portfolio's performance line (e.g. Performance %)
    and each individual ticker's performance over time.

    :param portfolio_df: DataFrame with at least ['Date', portfolio_col].
                         E.g. from Portfolio.calculate_portfolio_performance(),
                         where portfolio_col might be "Performance (%)" or just "Performance".
    :param all_tickers_perf: Dict of symbol -> ticker_performance_df,
                             each with ['Date', ticker_col] for that symbol's performance.
    :param portfolio_col: Column name in portfolio_df to plot, default 'Performance'.
    :param ticker_col: Column name in each ticker's DataFrame, default 'Performance'.
    :param portfolio_label: Legend label for the portfolio line, default 'Portfolio'.
    """
    fig = go.Figure()

    # 1) Add the portfolio line
    if portfolio_col in portfolio_df.columns:
        fig.add_trace(go.Scatter(
            x=portfolio_df['Date'],
            y=portfolio_df[portfolio_col],
            mode='lines',
            name=portfolio_label,
            line=dict(color='black', width=3)
        ))
    else:
        raise ValueError(f"Column '{portfolio_col}' not found in portfolio_df.")

    # 2) Add each ticker's performance line
    for symbol, df_ticker in all_tickers_perf.items():
        if ticker_col not in df_ticker.columns:
            # If the ticker DataFrame doesn't have the expected performance column, skip or warn
            print(f"Warning: '{ticker_col}' not found in ticker '{symbol}' DataFrame.")
            continue

        fig.add_trace(go.Scatter(
            x=df_ticker['Date'],
            y=df_ticker[ticker_col],
            mode='lines',
            name=str(symbol)  # Legend label is the ticker symbol
        ))

    # Layout settings
    fig.update_layout(
        title="Portfolio & Tickers Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Performance",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_portfolio_performance(portfolio_df, performance_col='Performance (%)'):
    """
    Plot a single line chart showing the portfolio's overall performance over time.

    :param portfolio_df: DataFrame with at least ['Date', performance_col].
    :param performance_col: The column representing portfolio performance (in % or in decimal).
    """
    # Validate columns
    if performance_col not in portfolio_df.columns:
        raise ValueError(f"Column '{performance_col}' not found in portfolio_df")

    fig = go.Figure()

    # Single line for portfolio performance
    fig.add_trace(go.Scatter(
        x=portfolio_df["Date"],
        y=portfolio_df[performance_col],
        mode='lines',
        name="Portfolio Performance",
        line=dict(color='blue', width=3)
    ))

    fig.update_layout(
        title="Portfolio Overall Performance Over Time",
        xaxis_title="Date",
        yaxis_title=performance_col,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


import plotly.graph_objects as go
from plotly.express.colors import qualitative as qc

def plot_performance(
    portfolio_df=None,
    items=None,
    date_col="Date",
    perf_col="Performance",
    portfolio_label="Portfolio",
    item_labels=None,
    title="Performance Over Time"
):
    """
    Generalized function to plot performance over time for:
      - A portfolio (single line)
      - Multiple items (e.g., tickers or benchmarks)

    :param portfolio_df: DataFrame for the portfolio with [date_col, perf_col] (optional).
    :param items: Dict of symbol -> DataFrame, each DataFrame must have [date_col, perf_col].
    :param date_col: Column name for the date (default: 'Date').
    :param perf_col: Column name for the performance metric (default: 'Performance').
    :param portfolio_label: Legend label for the portfolio line (if portfolio_df is provided).
    :param item_labels: List of labels for items in `items` (if provided). Defaults to item keys.
    :param title: Title for the chart (default: "Performance Over Time").
    """
    fig = go.Figure()

    # 1) Add the portfolio line, if provided
    if portfolio_df is not None:
        if date_col not in portfolio_df.columns or perf_col not in portfolio_df.columns:
            raise ValueError(f"Portfolio DataFrame must contain '{date_col}' and '{perf_col}' columns.")
        fig.add_trace(go.Scatter(
            x=portfolio_df[date_col],
            y=portfolio_df[perf_col],
            mode='lines',
            name=portfolio_label,
            line=dict(color='blue', width=3)
        ))

    # 2) Handle multiple items (e.g., tickers or benchmarks)
    if items is not None:
        # If no labels are provided, use the keys of the items dict
        if item_labels is None:
            item_labels = list(items.keys())

        if len(items) != len(item_labels):
            raise ValueError("Length of items and item_labels must match.")

        # Use a color cycle from Plotly's qualitative palette
        color_cycle = qc.Plotly  # or qc.D3, qc.Set1, etc.

        for i, (symbol, df_item) in enumerate(items.items()):
            label = item_labels[i]
            color = color_cycle[i % len(color_cycle)]

            if date_col not in df_item.columns or perf_col not in df_item.columns:
                print(f"Skipping {symbol}: missing '{date_col}' or '{perf_col}' in DataFrame.")
                continue

            fig.add_trace(go.Scatter(
                x=df_item[date_col],
                y=df_item[perf_col],
                mode='lines',
                name=label,
                line=dict(color=color, width=2)
            ))

    # 3) Final layout settings
    fig.update_layout(
        title=title,
        xaxis_title=date_col,
        yaxis_title=perf_col,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_tickers_performance(all_tickers_perf,
                             date_col='Date',
                             perf_col='Performance',
                             title="Tickers Performance Over Time"):
    """
    Plot multiple ticker performance lines on one figure (one line per ticker).

    :param all_tickers_perf: Dict of symbol -> DataFrame,
                             each DataFrame must have [date_col, perf_col].
    :param date_col: Column name for the date (default 'Date').
    :param perf_col: Column name for the ticker performance metric (default 'Performance').
    :param title: Title for the figure.
    """
    fig = go.Figure()

    for symbol, df_ticker in all_tickers_perf.items():
        # Validate columns
        if date_col not in df_ticker.columns or perf_col not in df_ticker.columns:
            print(f"Skipping {symbol}: missing '{date_col}' or '{perf_col}' in DataFrame.")
            continue

        fig.add_trace(go.Scatter(
            x=df_ticker[date_col],
            y=df_ticker[perf_col],
            mode='lines',
            name=symbol  # Legend label is the ticker
        ))

    fig.update_layout(
        title=title,
        xaxis_title=date_col,
        yaxis_title=perf_col,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_absolute_cumulative_return(portfolio_df):
    """
    Plots the absolute cumulative return (in currency) as a line chart.
    Assumes portfolio_df has a 'Cumulative Return (Abs)' column.
    """
    fig = px.line(portfolio_df, x="Date", y="Cumulative Return (Abs)",
                  title="Absolute Cumulative Return Over Time")
    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


def plot_top_holdings_concentration(allocation_df, top_n=5):
    """
    Show a bar chart of top N tickers by Market Value to visualize concentration risk.
    Assumes allocation_df has ['Ticker', 'Market Value'].
    """
    df = allocation_df.sort_values("Market Value", ascending=False).head(top_n)
    fig = px.bar(df, x='Ticker', y='Market Value',
                 title=f"Top {top_n} Holdings by Market Value",
                 color='Ticker',
                 text='Market Value')
    fig.update_traces(textposition='outside')
    fig.update_layout(template='plotly_white', hovermode='x unified', showlegend=False)
    return fig


def plot_asset_allocation_by_type(allocation_df, type_col='AssetType'):
    """
    Create a pie chart of asset allocation by the specified 'type_col' (e.g. AssetType, Sector, etc.).
    Expects allocation_df to have columns [type_col, 'Market Value'].
    """
    # Summarize by asset type
    df_type = allocation_df.groupby(type_col)['Market Value'].sum().reset_index()
    fig = px.pie(df_type, names=type_col, values='Market Value', title=f"Allocation by {type_col}")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template='plotly_white')
    return fig

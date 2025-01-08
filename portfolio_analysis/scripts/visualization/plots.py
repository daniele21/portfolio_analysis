import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



def create_transaction_annotated_line_chart(portfolio, transactions, ticker_data):
    """
    Create a line chart of portfolio performance with annotations for buy/sell events.
    """
    fig = go.Figure()

    # Line chart for portfolio value
    fig.add_trace(go.Scatter(
        x=portfolio['Date'],
        y=portfolio['Total Value'],
        mode='lines',
        name='Portfolio Value (€)',
        line=dict(color='blue', width=2)
    ))

    # Add buy/sell markers
    for ticker in transactions['Ticker'].unique():
        ticker_txns = transactions[transactions['Ticker'] == ticker]
        ticker_prices = ticker_data[ticker]

        # Process each transaction
        for _, txn in ticker_txns.iterrows():
            event = txn['Operation'].lower()
            txn_date = txn['Date']
            price = ticker_prices.loc[txn_date, 'Close'] if txn_date in ticker_prices.index else None

            if pd.notna(price):
                fig.add_trace(go.Scatter(
                    x=[txn_date],
                    y=[portfolio.loc[portfolio['Date'] == txn_date, 'Total Value'].values[0]],
                    mode='markers+text',
                    name=f"{event.capitalize()} {ticker}",
                    marker=dict(
                        size=10,
                        color='green' if event == 'buy' else 'red',
                        symbol='triangle-up' if event == 'buy' else 'triangle-down'
                    ),
                    text=f"{event.capitalize()} {ticker}",
                    textposition="top center"
                ))

    # Layout settings
    fig.update_layout(
        title="Portfolio Value Over Time with Transactions",
        xaxis_title="Date",
        yaxis_title="Total Value (€)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_close_price_with_transactions(ticker, portfolio_ticker_performance, transactions):
    """
    Plot the close price of a ticker with grouped Buy/Sell transaction markers under a single legend entry.
    """
    # Extract the portfolio performance data for the selected ticker
    if ticker not in portfolio_ticker_performance:
        raise ValueError(f"No performance data found for ticker {ticker}")

    df = portfolio_ticker_performance[ticker].copy()

    # Add transaction markers
    ticker_txns = transactions[transactions['Ticker'] == ticker]

    fig = go.Figure()

    # Plot close price
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price (€)',
        line=dict(color='blue', width=2)
    ))

    # Add buy/sell markers
    first_buy = True
    first_sell = True
    for _, txn in ticker_txns.iterrows():
        event = txn['Operation'].lower()
        txn_date = txn['Date']
        price = df.loc[df['Date'] == txn_date, 'Close'].iloc[0] if txn_date in df['Date'].values else None

        if pd.notna(price):
            fig.add_trace(go.Scatter(
                x=[txn_date],
                y=[price],
                mode='markers+text',
                name="Buy" if event == "buy" and first_buy else "Sell" if event == "sell" and first_sell else None,
                marker=dict(
                    size=10,
                    color='green' if event == 'buy' else 'red',
                    symbol='triangle-up' if event == 'buy' else 'triangle-down'
                ),
                text=f"{event.capitalize()}",
                textposition="top center",
                showlegend=first_buy if event == "buy" else first_sell
            ))
            # Ensure only the first "Buy" and "Sell" appear in the legend
            if event == "buy":
                first_buy = False
            elif event == "sell":
                first_sell = False

    # Layout settings
    fig.update_layout(
        title=f"Close Price and Transactions for {ticker}",
        xaxis_title="Date",
        yaxis=dict(title="Close Price (€)"),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_unrealized_gains(ticker, portfolio_ticker_performance, transactions):
    """
    Plot unrealized gains (€) and cumulative percentage of unrealized gains for a specific ticker,
    with Buy/Sell transaction markers grouped under one legend entry.
    """
    # Extract the portfolio performance data for the selected ticker
    if ticker not in portfolio_ticker_performance:
        raise ValueError(f"No performance data found for ticker {ticker}")

    df = portfolio_ticker_performance[ticker].copy()
    df = df[df['Quantity Held'] > 0]
    # Calculate cumulative percentage of unrealized gains
    df['Cumulative Unrealized Gain (%)'] = (df['Unrealized Gains'] / df['Cost Basis']) * 100

    # Add transaction markers
    ticker_txns = transactions[transactions['Ticker'] == ticker]

    fig = go.Figure()

    # Plot unrealized gains
    # fig.add_trace(go.Scatter(
    #     x=df['Date'],
    #     y=df['Unrealized Gains'],
    #     mode='lines',
    #     name='Holding Value (€)',
    #     line=dict(color='orange', width=2, dash='dash')
    # ))

    # Plot cumulative percentage of unrealized gains
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Cumulative Unrealized Gain (%)'],
        mode='lines',
        name='Performance',
        yaxis='y2',
        line=dict(color='green', width=2, dash='dot')
    ))

    # Add buy/sell markers
    first_buy = True
    first_sell = True
    for _, txn in ticker_txns.iterrows():
        event = txn['Operation'].lower()
        txn_date = txn['Date']
        unrealized_gain = df.loc[df['Date'] == txn_date, 'Unrealized Gains'].iloc[0] if txn_date in df['Date'].values else None

        if pd.notna(unrealized_gain):
            fig.add_trace(go.Scatter(
                x=[txn_date],
                y=[unrealized_gain],
                mode='markers+text',
                name="Buy" if event == "buy" and first_buy else "Sell" if event == "sell" and first_sell else None,
                marker=dict(
                    size=10,
                    color='green' if event == 'buy' else 'red',
                    symbol='triangle-up' if event == 'buy' else 'triangle-down'
                ),
                text=f"{event.capitalize()}",
                textposition="top center",
                showlegend=first_buy if event == "buy" else first_sell
            ))
            # Ensure only the first "Buy" and "Sell" appear in the legend
            if event == "buy":
                first_buy = False
            elif event == "sell":
                first_sell = False

    # Layout settings
    fig.update_layout(
        title=f"Unrealized Gains and Cumulative % for {ticker}",
        xaxis_title="Date",
        # yaxis=dict(title="Unrealized Gains (€)"),
        yaxis=dict(
            title="Cumulative Unrealized Gain (%)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig




def create_ticker_performance_chart(ticker, ticker_data, portfolio_ticker_performance, transactions):
    """
    Create a performance plot for a specific ticker in the portfolio.
    Includes close price, unrealized gains, and cumulative percentage of unrealized gains.
    """
    # Extract the portfolio performance data for the selected ticker
    if ticker not in portfolio_ticker_performance:
        raise ValueError(f"No performance data found for ticker {ticker}")

    df = portfolio_ticker_performance[ticker].copy()

    # Calculate cumulative percentage of unrealized gains
    df['Cumulative Unrealized Gain (%)'] = (df['Unrealized Gains'] / df['Cost Basis']) * 100

    # Add transaction markers
    ticker_txns = transactions[transactions['Ticker'] == ticker]

    fig = go.Figure()

    # Plot close price
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price (€)',
        line=dict(color='blue', width=2)
    ))

    # Plot unrealized gains
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Unrealized Gains'],
        mode='lines',
        name='Unrealized Gains (€)',
        yaxis='y2',
        line=dict(color='orange', width=2, dash='dash')
    ))

    # Plot cumulative percentage of unrealized gains
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Cumulative Unrealized Gain (%)'],
        mode='lines',
        name='Cumulative Unrealized Gain (%)',
        yaxis='y3',
        line=dict(color='green', width=2, dash='dot')
    ))

    # Add buy/sell markers
    for _, txn in ticker_txns.iterrows():
        event = txn['Operation'].lower()
        txn_date = txn['Date']
        price = df.loc[df['Date'] == txn_date, 'Close'].iloc[0] if txn_date in df['Date'].values else None

        if pd.notna(price):
            fig.add_trace(go.Scatter(
                x=[txn_date],
                y=[price],
                mode='markers+text',
                name=f"{event.capitalize()}",
                marker=dict(
                    size=10,
                    color='green' if event == 'buy' else 'red',
                    symbol='triangle-up' if event == 'buy' else 'triangle-down'
                ),
                text=f"{event.capitalize()}",
                textposition="top center"
            ))

    # Layout settings
    fig.update_layout(
        title=f"Performance of {ticker} in Portfolio",
        xaxis_title="Date",
        yaxis=dict(title="Close Price (€)"),
        yaxis2=dict(
            title="Unrealized Gains (€)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        yaxis3=dict(
            title="Cumulative Unrealized Gain (%)",
            overlaying="y",
            side="right",
            position=0.15,
            showgrid=False
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


# def create_ticker_performance_chart(ticker, ticker_data, transactions):
#     """
#     Create a performance plot for a specific ticker.
#     Includes close price, daily returns, and cumulative returns.
#     """
#     # Extract data for the selected ticker
#     df = ticker_data[ticker].copy()
#     df['Daily Return (%)'] = df['Close'].pct_change() * 100
#     df['Cumulative Return (%)'] = (1 + df['Close'].pct_change()).cumprod() - 1
#
#     # Add transaction markers
#     ticker_txns = transactions[transactions['Ticker'] == ticker]
#
#     fig = go.Figure()
#
#     # Plot close price
#     fig.add_trace(go.Scatter(
#         x=df.index,
#         y=df['Close'],
#         mode='lines',
#         name='Close Price (€)',
#         line=dict(color='blue', width=2)
#     ))
#
#     # Plot cumulative returns
#     fig.add_trace(go.Scatter(
#         x=df.index,
#         y=df['Cumulative Return (%)'] * 100,
#         mode='lines',
#         name='Cumulative Return (%)',
#         yaxis='y2',
#         line=dict(color='green', width=2, dash='dot')
#     ))
#
#     # Add buy/sell markers
#     for _, txn in ticker_txns.iterrows():
#         event = txn['Operation'].lower()
#         txn_date = txn['Date']
#         price = df.loc[txn_date, 'Close'] if txn_date in df.index else None
#
#         if pd.notna(price):
#             fig.add_trace(go.Scatter(
#                 x=[txn_date],
#                 y=[price],
#                 mode='markers+text',
#                 name=f"{event.capitalize()}",
#                 marker=dict(
#                     size=10,
#                     color='green' if event == 'buy' else 'red',
#                     symbol='triangle-up' if event == 'buy' else 'triangle-down'
#                 ),
#                 text=f"{event.capitalize()}",
#                 textposition="top center"
#             ))
#
#     # Layout settings
#     fig.update_layout(
#         title=f"Performance of {ticker}",
#         xaxis_title="Date",
#         yaxis=dict(title="Close Price (€)"),
#         yaxis2=dict(
#             title="Cumulative Return (%)",
#             overlaying="y",
#             side="right",
#             showgrid=False
#         ),
#         template="plotly_white",
#         hovermode="x unified",
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#     )
#
#     return fig


def create_advanced_line_chart(portfolio):
    """Enhanced line chart for portfolio value over time."""
    fig = go.Figure()

    # Line chart for Total Value
    fig.add_trace(go.Scatter(
        x=portfolio['Date'],
        y=portfolio['Total Value'],
        mode='lines+markers',
        name='Portfolio Value (€)',
        line=dict(color='blue', width=2),
        marker=dict(size=5)
    ))

    # Highlight maximum and minimum values
    max_value = portfolio['Total Value'].max()
    min_value = portfolio['Total Value'].min()
    max_date = portfolio.loc[portfolio['Total Value'].idxmax(), 'Date']
    min_date = portfolio.loc[portfolio['Total Value'].idxmin(), 'Date']

    fig.add_trace(go.Scatter(
        x=[max_date, min_date],
        y=[max_value, min_value],
        mode='markers+text',
        name='Extremes',
        text=["Max Value", "Min Value"],
        textposition="top center",
        marker=dict(size=10, color='red')
    ))

    # Layout settings
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Total Value (€)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_advanced_pie_chart(allocation):
    """Enhanced pie chart for current portfolio allocation."""
    fig = px.pie(
        allocation,
        names='Ticker',
        values='Market Value',
        title='Current Portfolio Allocation',
        color_discrete_sequence=px.colors.sequential.RdBu
    )

    # Customize layout
    fig.update_traces(
        textinfo='label+percent',
        hoverinfo='label+percent+value',
        marker=dict(line=dict(color='#000000', width=1.5))
    )
    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5)
    )

    return fig


def create_advanced_bar_chart(allocation):
    """Enhanced bar chart for realized and unrealized gains."""
    fig = go.Figure()

    # Realized Gains
    fig.add_trace(go.Bar(
        name='Realized Gains',
        x=allocation['Ticker'],
        y=allocation['Realized Gains'],
        text=allocation['Realized Gains'].apply(lambda x: f"€{x:,.2f}"),
        textposition='outside',
        marker_color='green'
    ))

    # Unrealized Gains
    fig.add_trace(go.Bar(
        name='Unrealized Gains',
        x=allocation['Ticker'],
        y=allocation['Unrealized Gains'],
        text=allocation['Unrealized Gains'].apply(lambda x: f"€{x:,.2f}"),
        textposition='outside',
        marker_color='blue'
    ))

    # Add reference line for significant gains
    threshold = 1000
    fig.add_trace(go.Scatter(
        x=allocation['Ticker'],
        y=[threshold] * len(allocation['Ticker']),
        mode='lines',
        name='Threshold (€1000)',
        line=dict(color='red', dash='dash')
    ))

    # Customize layout
    fig.update_layout(
        title="Realized and Unrealized Gains by Ticker",
        xaxis_title="Ticker",
        yaxis_title="Gains (€)",
        template="plotly_white",
        barmode='group',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_line_chart(portfolio):
    """Plot the portfolio value over time."""
    fig = px.line(portfolio, x='Date', y='Total Value', title='Portfolio Value Over Time')
    fig.update_layout(xaxis_title="Date", yaxis_title="Total Value (€)", hovermode="x unified")
    return fig


def create_pie_chart(allocation):
    """Plot the current portfolio allocation."""
    fig = px.pie(allocation, names='Ticker', values='Market Value', title='Current Portfolio Allocation')
    return fig


def create_bar_chart(allocation):
    """Plot realized and unrealized gains as a bar chart."""
    fig = go.Figure(data=[
        go.Bar(name='Value', x=allocation['Ticker'], y=allocation['Unrealized Gains'], marker_color='blue')
    ])
    fig.update_layout(
        title="Value by Ticker",
        xaxis_title="Ticker",
        yaxis_title="Amount (€)",
        barmode='group'
    )
    return fig

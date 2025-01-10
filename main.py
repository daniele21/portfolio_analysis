import streamlit as st
import pandas as pd
from datetime import datetime


from portfolio_analysis.scripts.data.portfolio import Portfolio, calculate_kpis
from portfolio_analysis.scripts.data.ticker import BenchmarkCollection, TickerCollection
from portfolio_analysis.scripts.visualization.plots import plot_performance_with_annotations, plot_performance, \
    create_pie_chart, plot_asset_allocation_by_type, plot_bar_realized_vs_unrealized



@st.cache_data
def read_data(tickers, start_date, end_date, transactions):
    ticker_collection = TickerCollection(tickers, start_date, end_date)
    ticker_collection.fetch_fundamental_data()
    all_data = ticker_collection.fetch_price_data()
    all_perfs = ticker_collection.calculate_all_ticker_performances()

    # Build Portfolio
    my_portfolio = Portfolio(
        name="Portfolio",
        transactions=transactions,
        tickers_map=ticker_collection.tickers_map
    )

    # Calculate portfolio-level performance
    # Suppose after computing portfolio performance:
    portfolio_df, all_tickers_perf = my_portfolio.calculate_portfolio_performance()

    # Also create an allocation DataFrame (final day or group by Ticker):
    allocation_df = pd.DataFrame()
    for sym, df_ in all_tickers_perf.items():
        if not df_.empty:
            last_row = df_.iloc[-1:].copy()
            allocation_df = pd.concat((allocation_df, last_row), ignore_index=True)

    # e.g. keep only relevant columns
    allocation_df = allocation_df[
        ['Ticker', 'Market Value', 'Unrealized Gains', 'Realized Gains', 'AssetType', 'Sector', 'Industry']]
    allocation_df.fillna(0, inplace=True)

    # Now create the plots:

    # fig_value = plot_portfolio_value_over_time(portfolio_df)
    # fig_value.show()

    benchmarks_tickers = ["^GSPC", "NDAQ"]
    benchmark_collection = BenchmarkCollection(benchmarks_tickers, start_date, end_date)

    # Fetch price data and compute cumulative returns
    benchmark_collection.fetch_fundamental_data()
    benchmark_collection.fetch_price_data()
    benchmark_cumulative_returns = benchmark_collection.compute_all_ticker_performances()
    benchmarks_labels = [df['title'].iloc[0] for ticker, df in benchmark_cumulative_returns.items()]
    benchmarks = {title: benchmark_cumulative_returns[label] for
                  title, label
                  in zip(benchmarks_labels, benchmarks_tickers)}

    return portfolio_df, all_tickers_perf, allocation_df, benchmarks
    # return portfolio_kpis, allocation_df, portfolio_performance, portfolio_ticker_performance, ticker_data, ticker_info


# Page Configuration
st.set_page_config(
    page_title="Portfolio Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Uploaded File
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "transactions" not in st.session_state:
    st.session_state["transactions"] = None

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Performance", "Allocation", "Transactions"])

# Home Tab
with tab1:
    st.markdown("""
        <div style="text-align: center;">
            <h1>Portfolio Analysis Tool</h1>
            <h3>Get detailed insights into your financial portfolio</h3>
            <hr style="border: 1px solid blue;">
        </div>
    """, unsafe_allow_html=True)

    st.info("Upload your transactions in CSV format to begin analyzing your portfolio.")
    uploaded_file = st.file_uploader("Upload your CSV file with transactions", type="csv")

    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        transactions = pd.read_csv(uploaded_file, parse_dates=['Date'])
        transactions['Date'] = pd.to_datetime(transactions['Date'], format="%d/%m/%Y")
        st.session_state["transactions"] = transactions
        st.success("File uploaded successfully!")
        st.write("Preview of uploaded transactions:")
        st.dataframe(transactions.head())
    elif not st.session_state["uploaded_file"]:
        st.warning("Please upload a valid CSV file to start.")

# Check if Transactions Are Available
transactions = st.session_state.get("transactions")
if transactions is not None:
    tickers = transactions['Ticker'].unique().tolist()
    start_date = transactions['Date'].min()
    end_date = datetime.today().date()

    portfolio_df, all_tickers_perf, allocation_df, benchmarks = read_data(tickers,
                                                                          start_date,
                                                                          end_date,
                                                                          transactions)
    portfolio_kpis = calculate_kpis(portfolio_df, allocation_df, all_tickers_perf)

# Performance Tab
with tab2:
    if transactions is not None:
        st.header("Portfolio Performance Overview")

        # Portfolio KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"€ {portfolio_kpis['value']:,.0f}")
        col2.metric("Unrealized Gains", f"€ {portfolio_kpis['unrealized_gains']:.2f}")
        col3.metric("Performance", f"{portfolio_kpis['performance']:.0f} %", help="Portfolio performance to date.")

        st.markdown("---")

        # Portfolio Value Over Time
        benchmarks_tickers = [df['title'].iloc[0] for x, df in benchmarks.items()]

        st.subheader("Portfolio Value Over Time")
        selected_benchmark_ticker = st.multiselect("Select Benchmark:", benchmarks_tickers)
        data_dict = {ticker: benchmarks[ticker] for ticker in selected_benchmark_ticker}
        # title = ' - '.join([value['title'].iloc[0] for _, value in data_dict.items()])

        annotated_line_chart = plot_performance(
                portfolio_df=portfolio_df,
                items=data_dict,
                # transactions=transactions_df,
                date_col="Date",
                perf_col="Performance (%)",  # or "Cumulative Return (%)"
                portfolio_label="My Portfolio",
                item_labels=selected_benchmark_ticker,
                title="Portfolio, Tickers, and Benchmarks Performance"
            )
        st.plotly_chart(annotated_line_chart)

        st.markdown("---")

        # Best and Worst Performing Tickers
        st.subheader("Best and Worst Performing Tickers")
        sub_col1, sub_col2 = st.columns(2)

        best_ticker = portfolio_kpis["best_ticker"]
        worst_ticker = portfolio_kpis["worst_ticker"]

        with sub_col1:
            st.metric("Best Performing Ticker", best_ticker['ticker'])
            st.metric("Value", f"€ {best_ticker['unrealized_gains']:.2f}")
            st.metric("Performance", f"{best_ticker['performance']:.2f} %")

        with sub_col2:
            st.metric("Worst Performing Ticker", worst_ticker['ticker'])
            st.metric("Value", f"€ {worst_ticker['unrealized_gains']:.2f}")
            st.metric("Performance", f"{worst_ticker['performance']:.2f} %", delta_color="inverse")

        st.markdown("---")

        # Unrealized Gains for a Selected Ticker
        st.subheader("Unrealized Gains for Selected Ticker")
        selected_ticker = st.multiselect("Select Ticker:", tickers)

        data_dict = {ticker: all_tickers_perf[ticker] for ticker in selected_ticker}
        titles = [value['title'].iloc[0] for _, value in data_dict.items()]

        if selected_ticker:
            perf_chart = plot_performance(
                portfolio_df=None,
                items=data_dict,
                color='orange',
                # transactions=transactions,
                date_col="Date",
                perf_col="Performance (%)",
                portfolio_label="Portfolio",
                item_labels=titles,
                title=f"Ticker Performance Over Time"
            )
            st.plotly_chart(perf_chart)

    else:
        st.warning("Please upload a CSV file in the Home tab to analyze performance.")

# Allocation Tab
with tab3:
    # Allocation Tab
    with tab3:
        if transactions is not None:
            st.header("Portfolio Allocation")

            pie_col1, pie_col2 = st.columns(2)
            with pie_col1:
                # Allocation by Ticker (Pie Chart)
                st.subheader("Allocation by Ticker")
                pie_chart = create_pie_chart(allocation_df)
                st.plotly_chart(pie_chart, use_container_width=True)

            with pie_col2:
                # Allocation by Asset Type (Pie Chart)
                st.subheader("Allocation by Asset Type")
                asset_type_chart = plot_asset_allocation_by_type(allocation_df)
                st.plotly_chart(asset_type_chart, use_container_width=True)

            # Unrealized vs Realized Gains (Bar Chart)
            st.subheader("Realized vs. Unrealized Gains")
            gains_bar_chart = plot_bar_realized_vs_unrealized(allocation_df)
            st.plotly_chart(gains_bar_chart, use_container_width=True)

        else:
            st.warning("Please upload a CSV file in the Home tab to view asset allocation.")

# Transactions Tab
with tab4:
    st.header("Uploaded Transactions")
    if transactions is not None:
        st.dataframe(transactions, use_container_width=True)
    else:
        st.warning("No transactions uploaded. Please upload a CSV file in the Home tab.")

# Footer
st.markdown("""
    <footer style="text-align: center; margin-top: 20px; font-size: small; color: #666;">
        <hr>
        <p>© 2025 Portfolio Analysis | All Rights Reserved</p>
    </footer>
""", unsafe_allow_html=True)

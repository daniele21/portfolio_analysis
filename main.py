import streamlit as st
import pandas as pd
from datetime import datetime
from portfolio_analysis.scripts.data.portfolio import PortfolioAnalysis, TickerAnalysis, calculate_kpis
from portfolio_analysis.scripts.visualization.plots import create_pie_chart, create_bar_chart, \
    plot_close_price_with_transactions, plot_unrealized_gains, create_transaction_annotated_line_chart


@st.cache_data
def read_data(tickers, start_date, end_date):
    ticker_analysis = TickerAnalysis(tickers, start_date, end_date)
    ticker_data = ticker_analysis.fetch_data()
    ticker_info = ticker_analysis.fetch_fundamental_data()

    portfolio_analysis = PortfolioAnalysis(transactions, ticker_data)
    portfolio_performance, portfolio_ticker_performance = portfolio_analysis.calculate_portfolio_performance()
    allocation_df = pd.DataFrame()
    for ticker, value in portfolio_ticker_performance.items():
        allocation_df = pd.concat((allocation_df, portfolio_ticker_performance[ticker].iloc[-1:]))
    allocation_df = allocation_df[['Ticker', 'Quantity Held', 'Market Value', 'Realized Gains', 'Unrealized Gains']]
    allocation_df = allocation_df.groupby('Ticker').sum().reset_index()
    portfolio_kpis = calculate_kpis(portfolio_performance,
                                    portfolio_ticker_performance,
                                    allocation_df)

    return portfolio_kpis, allocation_df, portfolio_performance, portfolio_ticker_performance, ticker_data, ticker_info


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

    portfolio_kpis, allocation_df, portfolio_performance, portfolio_ticker_performance, ticker_data, ticker_info = read_data(
        tickers, start_date, end_date)

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
        st.subheader("Portfolio Value Over Time")
        annotated_line_chart = create_transaction_annotated_line_chart(portfolio_performance, transactions, ticker_data)
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
        selected_ticker = st.selectbox("Select Ticker:", tickers)

        if selected_ticker:
            perf_chart = plot_unrealized_gains(selected_ticker, portfolio_ticker_performance, transactions)
            st.plotly_chart(perf_chart)

    else:
        st.warning("Please upload a CSV file in the Home tab to analyze performance.")

# Allocation Tab
with tab3:
    if transactions is not None:
        st.header("Portfolio Allocation")
        pie_chart = create_pie_chart(allocation_df)
        st.plotly_chart(pie_chart)

        st.subheader("Asset Allocation by Market Value")
        bar_chart = create_bar_chart(allocation_df)
        st.plotly_chart(bar_chart)
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

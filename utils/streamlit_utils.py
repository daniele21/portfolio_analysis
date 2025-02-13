from datetime import datetime

import pandas as pd
import streamlit as st

from portfolio_analysis.scripts.data.portfolio import Portfolio, calculate_kpis
from portfolio_analysis.scripts.data.ticker import BenchmarkCollection, TickerCollection
from portfolio_analysis.scripts.visualization.plots import plot_performance

COLUMN_CONFIG = {
    "Operation": st.column_config.SelectboxColumn(
        label="Operation",
        options=["Buy", "Sell"],  # Restrict options to Buy or Sell
        help="Choose whether the transaction is a Buy or Sell",
    ),
    "Date": st.column_config.DateColumn("Transaction Date", format="DD/MM/YY"),
    "Name": st.column_config.TextColumn("Name"),
    "Ticker": st.column_config.TextColumn("Ticker"),
    "Quantity": st.column_config.NumberColumn("Quantity", step=1),
}


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
        tickers_map=ticker_collection.tickers_map,
    )

    # Calculate portfolio-level performance
    # Suppose after computing portfolio performance:
    portfolio_df, all_tickers_perf = my_portfolio.calculate_portfolio_performance()

    # Also create an allocation DataFrame (final day or group by Ticker):
    allocation_df = my_portfolio.compute_asset_allocation()

    # Now create the plots:

    # fig_value = plot_portfolio_value_over_time(portfolio_df)
    # fig_value.show()

    benchmarks_tickers = ["^GSPC", "NDAQ"]
    benchmark_collection = BenchmarkCollection(benchmarks_tickers, start_date, end_date)

    # Fetch price data and compute cumulative returns
    benchmark_collection.fetch_fundamental_data()
    benchmark_collection.fetch_price_data()
    benchmark_cumulative_returns = (
        benchmark_collection.compute_all_ticker_performances()
    )
    benchmarks_labels = [
        df["title"].iloc[0] for ticker, df in benchmark_cumulative_returns.items()
    ]
    benchmarks = {
        title: benchmark_cumulative_returns[label]
        for title, label in zip(benchmarks_labels, benchmarks_tickers)
    }

    return my_portfolio, portfolio_df, all_tickers_perf, allocation_df, benchmarks


@st.cache_data
def read_transactions(uploaded_file):
    def detect_separator(file):
        """Detect the most common separator in the first line of the file."""
        first_line = file.readline().decode("utf-8")  # Read first line
        file.seek(0)  # Reset file pointer
        return "," if first_line.count(",") > first_line.count(";") else ";"

    print("> Reading Transaction")
    sep = detect_separator(uploaded_file)
    try:
        transactions = pd.read_csv(
            uploaded_file,
            parse_dates=["Date"],
            date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
            sep=sep,
        )
    except Exception as e:
        transactions = pd.read_csv(
            uploaded_file,
            parse_dates=["Date"],
            date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d"),
            sep=sep,
        )
    return transactions


@st.cache_resource
def loading_portfolio_data(transactions):
    print("> Reading Portfolio Data")
    tickers = transactions["Ticker"].unique().tolist()
    start_date = transactions["Date"].min()
    end_date = datetime.today().date()

    ticker_collection = TickerCollection(tickers, start_date, end_date)
    ticker_collection.fetch_fundamental_data()
    # all_data = ticker_collection.fetch_price_data()
    # all_perfs = ticker_collection.calculate_all_ticker_performances()

    # Build Portfolio
    my_portfolio = Portfolio(
        name="Portfolio",
        transactions=transactions,
        tickers_map=ticker_collection.tickers_map,
    )

    benchmarks_tickers = ["^GSPC", "^IXIC", "URTH", "^RUT", "^DJI", "^FTSE"]
    benchmark_collection = BenchmarkCollection(benchmarks_tickers, start_date, end_date)

    return my_portfolio, benchmark_collection


@st.cache_data
def _general_performance(portfolio_df):
    annotated_line_chart = plot_performance(
        portfolio_df=portfolio_df,
        items=None,
        # transactions=transactions_df,
        date_col="Date",
        perf_col="Performance (%)",  # or "Cumulative Return (%)"
        portfolio_label="My Portfolio",
        item_labels=None,
    )
    st.plotly_chart(annotated_line_chart, key="one")


def download_transactions(transactions_df, label="Download 📈", filename=None):
    csv = transactions_df.to_csv(index=False).encode("utf-8")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.download_button(
        label=label,
        data=csv,
        file_name=f"transactions_{timestamp}.csv" if filename is None else filename,
        mime="text/csv",
    )


@st.dialog("Upload/Fill your transactions", width="large")
def upload_data():
    # st.markdown(
    #     "<h2 style='text-align: center;'>Transactions Format</h2>",
    #     unsafe_allow_html=True
    # )
    st.markdown("Make sure your transactions file follows the right format")
    with st.expander("Transaction Format"):

        # Example dataframe structure
        example_data = pd.DataFrame(
            {
                "Operation": ["Buy", "Sell"],
                "Date": ["01/01/24", "15/02/24"],
                "Name": ["Apple", "Google"],
                "Ticker": ["AAPL", "GOOGL"],
                "Quantity": [10, 5],
            }
        )

        st.dataframe(
            example_data, column_config=COLUMN_CONFIG, use_container_width=True
        )
        download_transactions(
            example_data, label="Template 📃", filename="template_transactions.csv"
        )
        st.markdown(
            "*For better know the ticker, search it from [Yahoo Finance](https://www.finance.yahoo.com)*"
        )

    st.divider()
    st.markdown("<h3 style='text-align: center;'>Upload</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your CSV file with transactions", type="csv"
    )

    st.divider()
    st.markdown(
        "<h3 style='text-align: center;'>Fill here</h3>", unsafe_allow_html=True
    )
    blank_data = pd.DataFrame(
        {
            "Operation": ["Buy"],
            "Date": [pd.to_datetime("01/01/25")],
            "Ticker": ["Yahoo Ticker"],
            "Quantity": [1],
        }
    )
    st.markdown(
        "*For better know the ticker, search it from [Yahoo Finance](https://www.finance.yahoo.com)*"
    )
    transactions_df = st.data_editor(
        blank_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config=COLUMN_CONFIG,
    )

    st.markdown(
        "Please remember to download the transactions otherwise you will lose you it!"
    )
    download_transactions(transactions_df)

    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        transactions = read_transactions(uploaded_file)
        st.session_state["transactions"] = transactions
        st.session_state["show_upload_dialog"] = False
        st.rerun()


# @st.cache_data
def compute_performance_by_date(portfolio, benchmark_collection, start_date, end_date):
    data, ticker_data = portfolio.calculate_portfolio_performance_by_date(
        start_date, end_date
    )
    # portfolio_df, all_tickers_perf = portfolio.calculate_portfolio_performance_by_date(start_date, end_date)

    # Also create an allocation DataFrame (final day or group by Ticker):
    allocation_df = pd.DataFrame()
    for sym, df_ in ticker_data.items():
        if not df_.empty:
            last_row = df_.iloc[-1:].copy()
            allocation_df = pd.concat((allocation_df, last_row), ignore_index=True)

    # print(allocation_df.head())
    # st.dataframe(allocation_df.head())
    # return None, None, None, None
    # e.g. keep only relevant columns
    allocation_df = allocation_df[
        [
            "Ticker",
            "Market Value",
            "Unrealized Gains",
            "Realized Gains",
            "AssetType",
            "Sector",
            "Industry",
        ]
    ]
    allocation_df.fillna(0, inplace=True)

    # Now create the plots:

    # fig_value = plot_portfolio_value_over_time(portfolio_df)
    # fig_value.show()

    # Fetch price data and compute cumulative returns
    benchmark_collection.fetch_fundamental_data()
    benchmark_collection.fetch_price_data()
    benchmark_cumulative_returns = (
        benchmark_collection.calculate_all_ticker_performances_by_date()
    )
    benchmarks_labels = [
        df["title"].iloc[0] for ticker, df in benchmark_cumulative_returns.items()
    ]
    benchmarks = {
        title: benchmark_cumulative_returns[label]
        for title, label in zip(benchmarks_labels, benchmarks_tickers)
    }

    return data, ticker_data, allocation_df, benchmarks

    # return portfolio_df


def render_page():
    load_upload_action_menu()

    if "show_upload_dialog" not in st.session_state:
        st.session_state["show_upload_dialog"] = False

    # Open dialog only if the flag is set
    if st.session_state["show_upload_dialog"]:
        upload_data()

    else:
        uploaded_file = st.session_state.get("uploaded_file")
        transactions = st.session_state.get("transactions")

    if transactions is not None and uploaded_file is not None:
        tickers = transactions["Ticker"].unique().tolist()
        start_date = transactions["Date"].min()
        end_date = datetime.today().date()
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date

        my_portfolio, portfolio_df, all_tickers_perf, allocation_df, benchmarks = (
            read_data(tickers, start_date, end_date, transactions)
        )
        portfolio_kpis = calculate_kpis(
            my_portfolio, portfolio_df, allocation_df, all_tickers_perf
        )
        st.session_state["portfolio"] = my_portfolio

        return (
            my_portfolio,
            portfolio_df,
            all_tickers_perf,
            allocation_df,
            benchmarks,
            portfolio_kpis,
            tickers,
        )

    st.warning("No transactions uploaded. Please upload a CSV file on sidebar menu.")
    return None


def load_upload_action_menu():
    if st.session_state.get("transactions") is None:
        if st.sidebar.button("📄 Load Transactions"):
            st.session_state["show_upload_dialog"] = True
            upload_data()
    else:
        st.sidebar.success(
            "File uploaded successfully! Enjoy the demo, and remember to leave feedback.😊"
        )

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
    """Reads transactions from an uploaded CSV file, detecting separator and handling date formats."""

    def detect_separator(file):
        """Detect the most common separator in the first line of the file."""
        first_line = file.readline().decode("utf-8")  # Read first line
        file.seek(0)  # Reset file pointer
        return "," if first_line.count(",") > first_line.count(";") else ";"

    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV file first.")
        return None

    try:
        # Detect the separator (comma or semicolon)
        sep = detect_separator(uploaded_file)

        # Read the CSV without parsing dates first
        transactions = pd.read_csv(uploaded_file, sep=sep, dtype=str)

        # Ensure required columns exist
        required_columns = {"Operation", "Date", "Name", "Ticker", "Quantity"}
        missing_columns = required_columns - set(transactions.columns)
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            return None

        # Detect and fix date formats (Only DD/MM/YYYY and DD/MM/YY allowed)
        possible_formats = ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]
        invalid_dates = []

        def parse_date(date_str):
            """Try parsing the date with allowed formats."""
            for fmt in possible_formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            invalid_dates.append(date_str)  # Track invalid dates
            return None  # Return None if format not recognized

        # Apply date parsing and track invalid dates
        transactions["Date"] = transactions["Date"].astype(str).apply(parse_date)

        # Check if any invalid dates were found
        if invalid_dates:
            st.error(
                f"❌ **Invalid date format found!**\n"
                f"Allowed formats: **DD/MM/YYYY** or **DD/MM/YY**.\n"
                f"Errors in: {', '.join(invalid_dates)}"
            )
            return None

        # Convert dates to uniform format (DD/MM/YYYY)
        # transactions["Date"] = transactions["Date"].dt.strftime("%d/%m/%Y")
        transactions["Date"] = pd.to_datetime(transactions["Date"], format="%d/%m/%Y", errors="coerce")
        transactions['Quantity'] = pd.to_numeric(transactions["Quantity"], errors="coerce")
        transactions['Quantity'] = transactions['Quantity'].apply(lambda x: int(x))
        st.success("✅ Transactions successfully uploaded!")
        return transactions

    except pd.errors.ParserError as pe:
        st.error(f"⚠️ **CSV Parsing Error:** {pe}. Ensure your file follows the correct format.")
    except Exception as e:
        st.error(f"❌ **Unexpected Error:** {e}. Please check your file and try again.")

    return None



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


def download_transactions(transactions_df, label="↓ Download Here ↓", filename=None):
    csv = transactions_df.to_csv(index=False).encode("utf-8")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.download_button(
        label=label,
        data=csv,
        file_name=f"transactions_{timestamp}.csv" if filename is None else filename,
        mime="text/csv",
    )


@st.dialog("Add Your Investment Transactions", width="large")
def upload_data():
    # --- HEADER ---
    st.markdown("<h2 style='text-align: center;'>Upload or Enter Your Transactions</h2>", unsafe_allow_html=True)
    st.markdown("Make sure your transactions file follows the right format.")

    # --- TRANSACTION FORMAT ---
    with st.container():
        st.markdown("### Transaction Format")

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

        st.dataframe(example_data, use_container_width=True)

        # Download template button
        st.download_button(
            label="📥 Download Template",
            data=example_data.to_csv(index=False),
            file_name="template_transactions.csv",
            mime="text/csv",
        )

        st.markdown(
            "**Not sure about the ticker symbol?** 🔎 *Find the correct one by searching the company name on [Yahoo Finance](https://www.finance.yahoo.com)* before entering it here."
        )

    # --- SECTION DIVIDER ---
    st.markdown("<hr style='border: 2px solid #ddd;'>", unsafe_allow_html=True)

    # --- UPLOAD SECTION ---
    st.markdown("<h3 style='text-align: center;'>📂 Upload Your Transactions</h3>", unsafe_allow_html=True)
    st.markdown("Upload your CSV file with transactions.")

    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click Browse",
        type="csv",
        help="File should be in CSV format and follow the provided template."
    )

    st.markdown("<hr style='border: 2px solid #ddd;'>", unsafe_allow_html=True)

    # --- MANUAL ENTRY SECTION ---
    st.markdown("<h3 style='text-align: center;'>✍️ Manually Add Transactions</h3>", unsafe_allow_html=True)

    blank_data = pd.DataFrame(
        {
            "Operation": ["Buy"],
            "Date": ["01/01/25"],
            "Name": ["Ticker Name"],
            "Ticker": ["Yahoo Ticker"],
            "Quantity": [1],
        }
    )

    st.markdown(
        "**Not sure about the ticker symbol?** 🔎 *Find the correct one by searching the company name on [Yahoo Finance](https://www.finance.yahoo.com)* before entering it here."
    )

    transactions_df = st.data_editor(
        blank_data,
        num_rows="dynamic",
        use_container_width=True,
    )

    # --- REMINDER TO SAVE ---
    st.markdown(
        "⚠️ **Don't forget to download your transactions!** If you leave without saving, your data will be lost.",
        unsafe_allow_html=True
    )

    st.download_button(
        label="💾 Download Transactions",
        data=transactions_df.to_csv(index=False),
        file_name="transactions.csv",
        mime="text/csv",
    )

    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        transactions = read_transactions(uploaded_file)
        transactions["Date"] = pd.to_datetime(transactions["Date"], format="%d/%m/%Y", errors="coerce")
        st.session_state["transactions"] = transactions
        st.session_state["show_upload_dialog"] = False
        if transactions is not None:
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

        start_date = transactions["Date"].min().date()
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

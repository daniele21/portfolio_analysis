import streamlit as st
import pandas as pd
from datetime import datetime
from datetime import date

from portfolio_analysis.scripts.data.optimization import STRATEGIES, optimize_and_plot, optimize, TARGET_RETURN, \
    SAME_RISK
from portfolio_analysis.scripts.data.portfolio import Portfolio, calculate_kpis
from portfolio_analysis.scripts.data.ticker import BenchmarkCollection, TickerCollection
from portfolio_analysis.scripts.utils.utils import colorize
from portfolio_analysis.scripts.visualization.plots import plot_performance_with_annotations, plot_performance, \
    create_pie_chart, plot_asset_allocation_by_type, plot_bar_realized_vs_unrealized, plot_optimization

HOME = 'Home'
PERFORMANCE = 'Performance'
OPTIMIZATION = 'Optimization'
TRANSACTIONS = 'Transactions'

TABS = [HOME, PERFORMANCE, OPTIMIZATION, TRANSACTIONS]

COLUMN_CONFIG = {
        "Operation": st.column_config.SelectboxColumn(
            label="Operation",
            options=["Buy", "Sell"],  # Restrict options to Buy or Sell
            help="Choose whether the transaction is a Buy or Sell"
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
        tickers_map=ticker_collection.tickers_map
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
    benchmark_cumulative_returns = benchmark_collection.compute_all_ticker_performances()
    benchmarks_labels = [df['title'].iloc[0] for ticker, df in benchmark_cumulative_returns.items()]
    benchmarks = {title: benchmark_cumulative_returns[label] for
                  title, label
                  in zip(benchmarks_labels, benchmarks_tickers)}

    return my_portfolio, portfolio_df, all_tickers_perf, allocation_df, benchmarks


def update_performance(_my_portfolio, start_date, end_date):
    portfolio_df, all_tickers_perf = _my_portfolio.calculate_portfolio_performance_by_date(start_date, end_date)

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
    benchmark_cumulative_returns = benchmark_collection.compute_all_ticker_performances(start_date, end_date)
    benchmarks_labels = [df['title'].iloc[0] for ticker, df in benchmark_cumulative_returns.items()]
    benchmarks = {title: benchmark_cumulative_returns[label] for
                  title, label
                  in zip(benchmarks_labels, benchmarks_tickers)}

    portfolio_kpis = calculate_kpis(my_portfolio, portfolio_df, allocation_df, all_tickers_perf)

    return portfolio_df, all_tickers_perf, allocation_df, benchmarks, portfolio_kpis


@st.cache_resource
def loading_portfolio_data(transactions):
    print('> Reading Portfolio Data')
    tickers = transactions['Ticker'].unique().tolist()
    start_date = transactions['Date'].min()
    end_date = datetime.today().date()

    ticker_collection = TickerCollection(tickers, start_date, end_date)
    ticker_collection.fetch_fundamental_data()
    # all_data = ticker_collection.fetch_price_data()
    # all_perfs = ticker_collection.calculate_all_ticker_performances()

    # Build Portfolio
    my_portfolio = Portfolio(
        name="Portfolio",
        transactions=transactions,
        tickers_map=ticker_collection.tickers_map
    )

    benchmarks_tickers = ["^GSPC", "^IXIC", 'URTH', '^RUT', '^DJI', '^FTSE']
    benchmark_collection = BenchmarkCollection(benchmarks_tickers, start_date, end_date)

    return my_portfolio, benchmark_collection


# @st.cache_data
def compute_performance_by_date(portfolio, benchmark_collection, start_date, end_date):
    data, ticker_data = portfolio.calculate_portfolio_performance_by_date(start_date, end_date)
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
        ['Ticker', 'Market Value', 'Unrealized Gains', 'Realized Gains', 'AssetType', 'Sector', 'Industry']]
    allocation_df.fillna(0, inplace=True)

    # Now create the plots:

    # fig_value = plot_portfolio_value_over_time(portfolio_df)
    # fig_value.show()

    # Fetch price data and compute cumulative returns
    benchmark_collection.fetch_fundamental_data()
    benchmark_collection.fetch_price_data()
    benchmark_cumulative_returns = benchmark_collection.calculate_all_ticker_performances_by_date()
    benchmarks_labels = [df['title'].iloc[0] for ticker, df in benchmark_cumulative_returns.items()]
    benchmarks = {title: benchmark_cumulative_returns[label] for
                  title, label
                  in zip(benchmarks_labels, benchmarks_tickers)}

    return data, ticker_data, allocation_df, benchmarks

    # return portfolio_df


@st.cache_data
def read_transactions(uploaded_file):
    def detect_separator(file):
        """Detect the most common separator in the first line of the file."""
        first_line = file.readline().decode("utf-8")  # Read first line
        file.seek(0)  # Reset file pointer
        return "," if first_line.count(",") > first_line.count(";") else ";"

    print('> Reading Transaction')
    sep = detect_separator(uploaded_file)
    try:
        transactions = pd.read_csv(uploaded_file,
                                   parse_dates=['Date'],
                                   date_parser=lambda x: datetime.strptime(x, "%d/%m/%Y"),
                                   sep=sep)
    except Exception as e:
        transactions = pd.read_csv(uploaded_file,
                                   parse_dates=['Date'],
                                   date_parser=lambda x: datetime.strptime(x, "%d/%m/%y"),
                                   sep=sep)
    return transactions


def _home_allocation(allocation_df):
    st.write('**Allocation**')
    pills = st.pills(label=None, options=['Ticker', 'Full Name', 'Type'], default='Ticker')

    if pills == 'Ticker':
        # st.subheader("Allocation by Asset")
        pie_chart = create_pie_chart(allocation_df, x='Ticker')
        st.plotly_chart(pie_chart, use_container_width=True)
    if pills == 'Full Name':
        # st.subheader("Allocation by Asset")
        pie_chart = create_pie_chart(allocation_df, x='Title')
        st.plotly_chart(pie_chart, use_container_width=True)
    elif pills == 'Type':
        # st.subheader("Allocation by Asset")
        pie_chart = plot_asset_allocation_by_type(allocation_df)
        st.plotly_chart(pie_chart, use_container_width=True)


def _home_kpis_returns(portfolio_kpis):
    with st.container():
        with st.expander('Portfolio Returns'):
            col1, col2, col3 = st.columns(3)
            col1.write("**Daily Returns**")
            col1_text = f"€ {portfolio_kpis['returns']['daily']['abs']:.2f}"
            col1.markdown(
                f"<h2 style='color:{'red' if portfolio_kpis['returns']['daily']['abs'] < 0 else 'green'};'>{col1_text}</h2>",
                unsafe_allow_html=True
            )
            col1_text = f"{portfolio_kpis['returns']['daily']['pct']:.2f} %"
            col1.markdown(
                # f"<div style='text-align: center;'>"
                f"<div style='color:{'red' if portfolio_kpis['returns']['daily']['pct'] < 0 else 'green'}; font-size:20px; font-weight:bold;'>{col1_text}</div>",
                unsafe_allow_html=True
            )

            col2.write("**Weekly Returns**")
            col2_text = f"€ {portfolio_kpis['returns']['weekly']['abs']:.2f}"
            col2.markdown(
                f"<h2 style='color:{'red' if portfolio_kpis['returns']['daily']['abs'] < 0 else 'green'};'>{col2_text}</h2>",
                unsafe_allow_html=True
            )
            col2_text = f"{portfolio_kpis['returns']['weekly']['pct']:.2f} %"
            col2.markdown(
                # f"<div style='text-align: center;'>"
                f"<div style='color:{'red' if portfolio_kpis['returns']['weekly']['pct'] < 0 else 'green'}; font-size:20px; font-weight:bold;'>{col2_text}</div>",
                unsafe_allow_html=True
            )

            col3.write("**Monthly Returns**")
            col3_text = f"€ {portfolio_kpis['returns']['monthly']['abs']:.2f}"
            col3.markdown(
                f"<h2 style='color:{'red' if portfolio_kpis['returns']['daily']['abs'] < 0 else 'green'};'>{col3_text}</h2>",
                unsafe_allow_html=True
            )
            col3_text = f"{portfolio_kpis['returns']['monthly']['pct']:.2f} %"
            col3.markdown(
                # f"<div style='text-align: center;'>"
                f"<div style='color:{'red' if portfolio_kpis['returns']['monthly']['pct'] < 0 else 'green'}; font-size:20px; font-weight:bold;'>{col3_text}</div>",
                unsafe_allow_html=True
            )

            # col1.metric("Daily Returns",
            #             f"€ {portfolio_kpis['returns']['daily']['abs']:.2f}",
            #             delta=f"{portfolio_kpis['returns']['daily']['pct']:.2f} %",
            #             border=False)
            # col2.metric("Weekly Returns",
            #             f"€ {portfolio_kpis['returns']['weekly']['abs']:.2f}",
            #             delta=f"{portfolio_kpis['returns']['weekly']['pct']:.2f} %",
            #             border=False)
            # col3.metric("Monthly Returns",
            #             f"€ {portfolio_kpis['returns']['monthly']['abs']:.2f}",
            #             delta=f"{portfolio_kpis['returns']['monthly']['pct']:.2f} %",
            #             border=False)


def _home_kpis_ticker(portfolio_kpis):
    with st.container():
        with st.expander('Best/Worst Assets'):
            best_ticker = portfolio_kpis["best_ticker"]
            worst_ticker = portfolio_kpis["worst_ticker"]
            col1, col2, col3 = st.columns(3)

            col1.write("**Best Performing Ticker**")
            col1_text = f"{best_ticker['ticker']}"
            col1.markdown(
                f"<h3>{col1_text}</h3>",
                unsafe_allow_html=True
            )
            col2.write('**Value**')
            col2_text = f"€ {best_ticker['unrealized_gains']:.2f}"
            col2.markdown(
                f"<h3>{col2_text}</h3>",
                unsafe_allow_html=True
            )
            col3.write('**Performance**')
            col3_text = f"{best_ticker['performance']:.2f} %"
            col3.markdown(
                f"<h3 style='color:{'red' if best_ticker['performance'] < 0 else 'green'};'>{col3_text}</h3>",
                unsafe_allow_html=True
            )

            col1.divider()
            col2.divider()
            col3.divider()

            col1.write("**Worst Performing Ticker**")
            col1_text = f"{worst_ticker['ticker']}"
            col1.markdown(
                f"<h3>{col1_text}</h3>",
                unsafe_allow_html=True
            )
            col2.write('**Value**')
            col2_text = f"€ {worst_ticker['unrealized_gains']:.2f}"
            col2.markdown(
                f"<h3>{col2_text}</h3>",
                unsafe_allow_html=True
            )
            col3.write('**Performance**')
            col3_text = f"{worst_ticker['performance']:.2f} %"
            col3.markdown(
                f"<h3 style='color:{'red' if worst_ticker['performance'] < 0 else 'green'};'>{col3_text}</h3>",
                unsafe_allow_html=True
            )


def _kpis(portfolio_kpis):
    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.subheader(f"**Total Value**")
        st.header(f"€ {portfolio_kpis['value']:,.0f}")
    with cols[1]:
        st.subheader(f"Gain/Loss")
        st.header(f"€ {portfolio_kpis['unrealized_gains']:.2f}")
    with cols[2]:
        st.subheader(f"Performance")
        st.header(f"{portfolio_kpis['performance']:.2f} %")

    st.divider()


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
    st.plotly_chart(annotated_line_chart, key='one')


def render_home_tab(allocation_df, portfolio_kpis, portfolio_df):
    transactions = st.session_state.get("transactions")
    if transactions is not None:
        col1, _, col2 = st.columns([2, 0.1, 1.2])
        with col1:
            st.header("Portfolio")
            _kpis(portfolio_kpis)
            _home_kpis_returns(portfolio_kpis)
            _general_performance(portfolio_df)
        with col2:
            st.header("Assets")
            _home_kpis_ticker(portfolio_kpis)
            st.divider()
            _home_allocation(allocation_df)

    else:
        st.warning("Please upload a CSV file in the Home tab to view asset allocation.")


def _portfolio_performance(portfolio_kpis, cur_vol, data_dict, portfolio_df, selected_benchmark_ticker):
    col1, col2, col3, col4 = st.columns(4)

    col1.subheader('**Total Value**')
    col1.header(f"€ {portfolio_kpis['value']:,.0f}")
    col2.subheader('**Gain/Loss**')
    col2_text = f"€ {portfolio_kpis['unrealized_gains']:.2f}"
    col2.markdown(
        f"<h2 style='color:{'red' if portfolio_kpis['unrealized_gains'] < 0 else 'green'};'>{col2_text}</h2>",
        unsafe_allow_html=True
    )
    col3.subheader('**Performance**')
    col3_text = f"{portfolio_kpis['performance']:.0f} %"
    col3.markdown(
        f"<h2 style='color:{'red' if portfolio_kpis['performance'] < 0 else 'green'};'>{col3_text}</h2>",
        unsafe_allow_html=True
    )
    col4.subheader('**Volatility**')
    col4_text = f'{cur_vol * 100:.0f} %'
    col4.markdown(
        f"<h2 >{col4_text}</h2>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    annotated_line_chart = plot_performance(
        portfolio_df=portfolio_df,
        items=data_dict,
        # transactions=transactions_df,
        date_col="Date",
        perf_col="Performance (%)",  # or "Cumulative Return (%)"
        portfolio_label="My Portfolio",
        item_labels=selected_benchmark_ticker,
        # title="Portfolio, Tickers, and Benchmarks Performance"
    )
    st.plotly_chart(annotated_line_chart)

    st.markdown("---")


def _asset_performance(portfolio_kpis, cur_vol, data_dict, portfolio_df, selected_benchmark_ticker, all_tickers_perf):
    selected_ticker = st.multiselect("Select Ticker:", tickers, default=tickers[0])
    ticker_data_dict = {ticker: all_tickers_perf[ticker] for ticker in selected_ticker}
    titles = [value['title'].iloc[0] for _, value in ticker_data_dict.items()]
    # st.write(data_dict['EVISO.MI'])

    final_data_dict = {**ticker_data_dict, **data_dict}
    final_titles = titles + selected_benchmark_ticker
    # st.write(final_titles)

    if selected_ticker:
        perf_chart = plot_performance(
            portfolio_df=None,
            items=final_data_dict,
            color='orange',
            # transactions=transactions,
            date_col="Date",
            perf_col="Performance (%)",
            portfolio_label="Portfolio",
            item_labels=final_titles,
            # title=f"Ticker Performance Over Time"
        )
        st.plotly_chart(perf_chart)

    # with cols[2]:
    col0, col1, col2, col3, col4 = st.columns(5)
    col0.subheader('**Asset**')
    col1.subheader('**Total Value**')
    col2.subheader('**Gain/Loss**')
    col3.subheader('**Performance**')
    col4.subheader('**Volatility**')
    for t in selected_ticker:
        df = ticker_data_dict[t].iloc[-1]
        vol = (ticker_data_dict[t]["Daily Return (%)"] / 100).std() * (252 ** 0.5)

        col0.header(f"**{df['Ticker']}**")
        col1.header(f"€ {df['Market Value']:,.0f}")
        col2_text = f"€ {df['Unrealized Gains']:.2f}"
        col2.markdown(
            f"<h2 style='color:{'red' if df['Unrealized Gains'] < 0 else 'green'};'>{col2_text}</h2>",
            unsafe_allow_html=True
        )
        col3_text = f"{df['Performance (%)']:.0f} %"
        col3.markdown(
            f"<h2 style='color:{'red' if df['Performance (%)'] < 0 else 'green'};'>{col3_text}</h2>",
            unsafe_allow_html=True
        )
        col4_text = f'{vol * 100:.0f} %'
        col4.markdown(
            f"<h2 >{col4_text}</h2>",
            unsafe_allow_html=True
        )

        # col1, col2, col3, _ = st.columns([1, 1, 1, 2])
        # with col1:
        #     st.metric("Best Performing Ticker", best_ticker['ticker'])
        #     st.metric("Worst Performing Ticker", worst_ticker['ticker'])
        #
        # with col2:
        #     st.metric("Value", f"€ {best_ticker['unrealized_gains']:.2f}")
        #     st.metric("Value", f"€ {worst_ticker['unrealized_gains']:.2f}")
        #
        # with col3:
        #     st.metric("Performance", f"{best_ticker['performance']:.2f} %")
        #     st.metric("Performance", f"{worst_ticker['performance']:.2f} %")



def render_performance_tab():
    transactions = st.session_state.get("transactions")
    min_date = st.session_state.get('start_date')
    max_date = st.session_state.get('end_date')
    my_portfolio = st.session_state.get('portfolio')
    today = date.today()

    if transactions is not None:
        header_cols = st.columns(2)

        with header_cols[0]:
            options = ["Portfolio", "Assets"]
            st.subheader('Choose the target analysis')
            selection = st.pills(label=None, options=options, selection_mode="single", default=options[0])
        start_date, end_date = timeframe_input(min_date, max_date, header_cols[0])

        portfolio_df, all_tickers_perf, allocation_df, benchmarks, portfolio_kpis = update_performance(my_portfolio,
                                                                                                       start_date,
                                                                                                       end_date)
        portfolio_df = portfolio_df[(portfolio_df['Date'].dt.date >= start_date) & \
                                    (portfolio_df['Date'].dt.date <= end_date)]
        all_tickers_perf = {ticker: all_tickers_perf[ticker][(all_tickers_perf[ticker]['Date'].dt.date >= start_date) & \
                                                             (all_tickers_perf[ticker]['Date'].dt.date <= end_date)] for ticker in all_tickers_perf}

        _, _, cur_vol = my_portfolio.current_weights()
        benchmarks_tickers = [df['title'].iloc[0] for x, df in benchmarks.items()]

        with header_cols[1]:
            st.subheader('Choose the Benchmark to compare')
            selected_benchmark_ticker = st.multiselect("Select Benchmark:", benchmarks_tickers)
            data_dict = {ticker: benchmarks[ticker][(benchmarks[ticker]['Date'].dt.date >= start_date) &\
                                                    (benchmarks[ticker]['Date'].dt.date <= end_date)] for ticker in selected_benchmark_ticker}

        st.divider()

        if selection == options[0]:
            st.subheader("Portfolio Performance Over Time")
            _portfolio_performance(portfolio_kpis, cur_vol, data_dict, portfolio_df, selected_benchmark_ticker)

        elif selection == options[1]:
            st.subheader("Asset Performance Over Time")
            _asset_performance(portfolio_kpis, cur_vol, data_dict, portfolio_df, selected_benchmark_ticker, all_tickers_perf)


    else:
        st.warning("Please upload a CSV file in the Home tab to analyze performance.")


def timeframe_input(min_date, max_date, container=None):
    today = date.today()
    default_start_date = min_date
    default_end_date = max_date

    container = st.columns(1)[0] if container is None else container

    with container:
        with st.form('Configure Date Range', border=False):
            col1, col2, col3 = st.columns([1, 1, 2], gap='medium')
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=default_start_date,
                    min_value=min_date,
                    max_value=today,  # Latest selectable date
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=default_end_date,
                    min_value=min_date,  # End date cannot be earlier than start date
                    max_value=today,
                )
            with col3:
                st.text('')
                submitted = st.form_submit_button("Submit")
                if submitted:
                    return start_date, end_date
                else:
                    # return  pd.Timestamp(min_date),  pd.Timestamp(max_date)
                    return pd.to_datetime(min_date).date(), pd.to_datetime(max_date).date()


def download_transactions(transactions_df,
                          label="Download 📈",
                          filename=None):
    csv = transactions_df.to_csv(index=False).encode("utf-8")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.download_button(
        label=label,
        data=csv,
        file_name=f"transactions_{timestamp}.csv" if filename is None else filename,
        mime="text/csv"
    )


@st.dialog('Upload/Fill your transactions', width='large')
def upload_data():
    # st.markdown(
    #     "<h2 style='text-align: center;'>Transactions Format</h2>",
    #     unsafe_allow_html=True
    # )
    st.markdown("Make sure your transactions file follows the right format")
    with st.expander('Transaction Format'):

        # Example dataframe structure
        example_data = pd.DataFrame({
            "Operation": ["Buy", "Sell"],
            "Date": ["01/01/24", "15/02/24"],
            "Name": ['Apple', 'Google'],
            "Ticker": ["AAPL", "GOOGL"],
            "Quantity": [10, 5]
        })

        st.dataframe(example_data,
                     column_config=COLUMN_CONFIG,
                     use_container_width=True)
        download_transactions(example_data,
                              label='Template 📃',
                              filename='template_transactions.csv')
        st.markdown("*For better know the ticker, search it from [Yahoo Finance](https://www.finance.yahoo.com)*")

    st.divider()
    st.markdown(
        "<h3 style='text-align: center;'>Upload</h3>",
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Upload your CSV file with transactions", type="csv")

    st.divider()
    st.markdown(
        "<h3 style='text-align: center;'>Fill here</h3>",
        unsafe_allow_html=True
    )
    blank_data = pd.DataFrame({
        "Operation": ["Buy"],
        "Date": [pd.to_datetime("01/01/25")],
        "Ticker": ["Yahoo Ticker"],
        "Quantity": [1]
    })
    st.markdown("*For better know the ticker, search it from [Yahoo Finance](https://www.finance.yahoo.com)*")
    transactions_df = st.data_editor(blank_data,
                                     num_rows="dynamic",
                                     use_container_width=True,
                                     column_config=COLUMN_CONFIG)

    st.markdown("Please remember to download the transactions otherwise you will lose you it!")
    download_transactions(transactions_df)

    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        transactions = read_transactions(uploaded_file)
        st.session_state["transactions"] = transactions
        st.session_state["show_upload_dialog"] = False
        st.rerun()


def switch_tab(tab_name):
    if st.session_state["current_tab"] != tab_name:
        st.session_state["current_tab"] = tab_name
        st.rerun()


def render_optimization_tab(my_portfolio, allocation_df, portfolio_kpis, portfolio_df):
    if my_portfolio is not None and \
            allocation_df is not None and \
            portfolio_kpis is not None and \
            portfolio_df is not None:
        st.header("Portfolio Optimization")

        target_return = None
        target_volatility = None
        risk_free_rate = 0.02
        curr_weights, cur_ret, cur_vol = my_portfolio.current_weights()
        curr_weights_sorted = curr_weights['Allocation(%)'].sort_values(ascending=False) \
            .to_frame() \
            .T \
            .round(0)
        _, mean_returns, cov_matrix = my_portfolio.get_returns_matrix()
        df_random, frontiers, port_opt = optimize(mean_returns,
                                                  cov_matrix,
                                                  risk_free_rate=risk_free_rate,
                                                  target_return=target_return,
                                                  target_volatility=target_volatility
                                                  )

        cols = st.columns([1, 0.2, 1])
        with cols[0]:
            st.write('**Current Portfolio Allocation**')
            st.dataframe(curr_weights_sorted, use_container_width=True)

            fig = plot_optimization(df_random, frontiers, port_opt, (cur_ret, cur_vol))
            st.plotly_chart(fig, use_container_width=True)
        with cols[2]:
            pills = st.pills("Choose your Strategy", options=STRATEGIES, default=STRATEGIES[0])
            target_disabled = pills != TARGET_RETURN
            risk_disabled = pills != SAME_RISK

            input_cols = st.columns(2)
            with input_cols[0]:
                user_input_tr = st.number_input("Desired Annual Return (%)",
                                                value=5.0,
                                                step=1.0,
                                                disabled=target_disabled)
                target_return = user_input_tr / 100.0
            with input_cols[1]:
                user_input_tv = st.number_input("Desired Volatility (%)",
                                                value=15.0,
                                                step=1.0,
                                                disabled=risk_disabled)
                target_volatility = user_input_tv / 100.0

            df_random, frontiers, port_opt = optimize(mean_returns,
                                                      cov_matrix,
                                                      risk_free_rate=risk_free_rate,
                                                      target_return=target_return,
                                                      target_volatility=target_volatility
                                                      )
            st.divider()
            space = st.columns(3)
            space[0].metric(label='Expected Annual Return',
                            value=f"{port_opt[pills]['ret'] * 100:.2f} %")
            space[1].metric(label='Annual Volatility',
                            value=f"{port_opt[pills]['vol'] * 100:.2f} %")
            space[2].metric(label='Sharpe Ratio',
                            value=f"{port_opt[pills]['sharpe']:.2f}")
            st.divider()
        with cols[2]:
            st.write(f'**New Allocation** following **{pills} strategy**')
            opt_weights = pd.DataFrame(port_opt[pills]['weights'], index=['%'])
            opt_allocation = (opt_weights * 100).round(0)
            st.dataframe(opt_allocation, use_container_width=True)
            st.divider()

            st.write(f'**Allocation Difference from Current Portfolio**')
            diff_curr_weights = curr_weights.rename(columns={'Allocation(%)': '%'})['%'].sort_index() / 100
            # st.dataframe(diff_curr_weights.to_frame().T, use_container_width=True)
            diff_opt_weights = opt_weights.T.sort_index()
            # st.dataframe(diff_opt_weights.T, use_container_width=True)
            diff_weights = ((diff_opt_weights.T - diff_curr_weights.T) * 100)
            diff_weights = diff_weights.style.applymap(colorize) \
                .format("{:.0f}") \
                .set_properties(**{'font-size': '24pt'})

            st.dataframe(diff_weights, use_container_width=True)
            # st.text('Green coloured values -> ')

def render_transaction_tab():
    def change_transactions():
        st.session_state['transactions'] = st.session_state['edited_transactions']

    transactions = st.session_state.get('transactions')
    st.header("Current Transactions")
    if st.button("📄 Load Transactions"):
        st.session_state["show_upload_dialog"] = True
        st.rerun()

    if transactions is not None:
        st.markdown("You can **Add** | **Edit** | **Remove** the transactions:")
        st.session_state["edited_transactions"] = transactions.copy()
        edited_transactions = st.data_editor(st.session_state["edited_transactions"],
                                             use_container_width=True,
                                             num_rows='dynamic',
                                             column_config=COLUMN_CONFIG)
        st.session_state["edited_transactions"] = edited_transactions
        st.button('Submit changes', on_click=change_transactions)
        download_transactions(edited_transactions)
    else:
        st.warning("No transactions uploaded. Please upload a CSV file in the Home tab.")


if __name__ == '__main__':

    st.set_page_config(
        page_title="My Financial Vision",
        layout="wide",
        initial_sidebar_state="auto"
    )
    st.session_state['loading_data'] = False
    st.markdown("""
            <div style="text-align: center;">
                <h1>💸 My Financial Vision 💸</h1>
                <h3>Own Your Finances, Own Your Future 🔒</h3>
                <hr style="border: 1px solid blue;">
            </div>
        """, unsafe_allow_html=True)
    uploaded_file = None
    transactions = None
    allocation_df, portfolio_kpis, portfolio_df, my_portfolio = None, None, None, None

    if "show_upload_dialog" not in st.session_state:
        st.session_state["show_upload_dialog"] = False

    # Open dialog only if the flag is set
    if st.session_state["show_upload_dialog"]:
        upload_data()

    else:
        uploaded_file = st.session_state.get("uploaded_file")
        transactions = st.session_state.get("transactions")

    # home_tab, perf_tab, optimization_tab, transaction_tab = st.tabs(TABS)
    selected_tab = st.pills(
        "Navigate",
        options=TABS,
        selection_mode="single",
        default=TRANSACTIONS
    )

    if st.session_state.get("current_tab") is None:
        st.session_state["current_tab"] = HOME

    if transactions is not None and uploaded_file is not None:
        # st.info('Loading Data...')
        tickers = transactions['Ticker'].unique().tolist()
        start_date = transactions['Date'].min()
        end_date = datetime.today().date()
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date

        my_portfolio, portfolio_df, all_tickers_perf, allocation_df, benchmarks = read_data(tickers,
                                                                                            start_date,
                                                                                            end_date,
                                                                                            transactions)
        portfolio_kpis = calculate_kpis(my_portfolio, portfolio_df, allocation_df, all_tickers_perf)
        # st.write(all_tickers_perf)
        st.session_state['portfolio'] = my_portfolio

    if selected_tab == HOME:
        switch_tab(HOME)
        if my_portfolio is not None and \
                allocation_df is not None and \
                portfolio_kpis is not None and \
                portfolio_df is not None:
            render_home_tab(allocation_df, portfolio_kpis, portfolio_df)
        else:
            st.warning("No transactions uploaded. Please upload a CSV file")

    if selected_tab == PERFORMANCE:
        switch_tab(PERFORMANCE)
        render_performance_tab()

    # Allocation Tab
    if selected_tab == OPTIMIZATION:
        switch_tab(OPTIMIZATION)
        render_optimization_tab(my_portfolio, allocation_df, portfolio_kpis, portfolio_df)

    # Transactions Tab
    if selected_tab == TRANSACTIONS:
        switch_tab(TRANSACTIONS)
        render_transaction_tab()

    # Footer
    st.markdown("""
        <footer style="text-align: center; margin-top: 20px; font-size: small; color: #666;">
            <hr>
            <p>© 2025 Portfolio Analysis | All Rights Reserved</p>
        </footer>
    """, unsafe_allow_html=True)

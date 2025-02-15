from datetime import date, datetime

import pandas as pd
import streamlit as st

from portfolio_analysis.scripts.data.portfolio import calculate_kpis
from portfolio_analysis.scripts.data.ticker import BenchmarkCollection
from portfolio_analysis.scripts.visualization.plots import plot_performance
from portfolio_analysis.scripts.constants.color import GREEN, RED, ORANGE
from utils.streamlit_utils import read_data, render_page, upload_data


def render_performance_tab():
    transactions = st.session_state.get("transactions")
    min_date = st.session_state.get("start_date")
    max_date = st.session_state.get("end_date")
    my_portfolio = st.session_state.get("portfolio")
    today = date.today()

    if transactions is not None:
        header_cols = st.columns(2)

        with header_cols[0]:
            options = ["Portfolio", "Assets"]
            st.subheader("Choose the target analysis")
            selection = st.pills(
                label=None, options=options, selection_mode="single", default=options[0]
            )
        start_date, end_date = timeframe_input(min_date, max_date, header_cols[0])

        portfolio_df, all_tickers_perf, allocation_df, benchmarks, portfolio_kpis = (
            update_performance(my_portfolio, start_date, end_date)
        )
        portfolio_df = portfolio_df[
            (portfolio_df["Date"].dt.date >= start_date)
            & (portfolio_df["Date"].dt.date <= end_date)
        ]
        all_tickers_perf = {
            ticker: all_tickers_perf[ticker][
                (all_tickers_perf[ticker]["Date"].dt.date >= start_date)
                & (all_tickers_perf[ticker]["Date"].dt.date <= end_date)
            ]
            for ticker in all_tickers_perf
        }

        _, _, cur_vol = my_portfolio.current_weights()
        benchmarks_tickers = [df["title"].iloc[0] for x, df in benchmarks.items()]

        with header_cols[1]:
            st.subheader("Choose the Benchmark to compare")
            selected_benchmark_ticker = st.multiselect(
                "Select Benchmark:", benchmarks_tickers
            )
            data_dict = {
                ticker: benchmarks[ticker][
                    (benchmarks[ticker]["Date"].dt.date >= start_date)
                    & (benchmarks[ticker]["Date"].dt.date <= end_date)
                ]
                for ticker in selected_benchmark_ticker
            }

        st.divider()

        if selection == options[0]:
            st.subheader("Portfolio Performance Over Time")
            _portfolio_performance(
                portfolio_kpis,
                cur_vol,
                data_dict,
                portfolio_df,
                selected_benchmark_ticker,
            )

        elif selection == options[1]:
            st.subheader("Asset Performance Over Time")
            _asset_performance(
                portfolio_kpis,
                cur_vol,
                data_dict,
                portfolio_df,
                selected_benchmark_ticker,
                all_tickers_perf,
            )


def update_performance(_my_portfolio, start_date, end_date):
    portfolio_df, all_tickers_perf = (
        _my_portfolio.calculate_portfolio_performance_by_date(start_date, end_date)
    )

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
    benchmark_cumulative_returns = benchmark_collection.compute_all_ticker_performances(
        start_date, end_date
    )
    benchmarks_labels = [
        df["title"].iloc[0] for ticker, df in benchmark_cumulative_returns.items()
    ]
    benchmarks = {
        title: benchmark_cumulative_returns[label]
        for title, label in zip(benchmarks_labels, benchmarks_tickers)
    }

    portfolio_kpis = calculate_kpis(
        my_portfolio, portfolio_df, allocation_df, all_tickers_perf
    )

    return portfolio_df, all_tickers_perf, allocation_df, benchmarks, portfolio_kpis


def timeframe_input(min_date, max_date, container=None):
    today = date.today()
    default_start_date = min_date
    default_end_date = max_date

    container = st.columns(1)[0] if container is None else container

    with container:
        with st.form("Configure Date Range", border=False):
            col1, col2, col3 = st.columns([1, 1, 2], gap="medium")
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
                st.text("")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    return start_date, end_date
                else:
                    # return  pd.Timestamp(min_date),  pd.Timestamp(max_date)
                    return (
                        pd.to_datetime(min_date).date(),
                        pd.to_datetime(max_date).date(),
                    )


def _portfolio_performance(
    portfolio_kpis, cur_vol, data_dict, portfolio_df, selected_benchmark_ticker
):
    col1, col2, col3, col4 = st.columns(4)

    col1.subheader("**Total Value**")
    col1.header(f"€ {portfolio_kpis['value']:,.0f}")
    col2.subheader("**Gain/Loss**")
    col2_text = f"€ {portfolio_kpis['unrealized_gains']:.2f}"
    col2.markdown(
        f"<h2 style='color:{f'{RED}' if portfolio_kpis['unrealized_gains'] < 0 else f'{GREEN}'};'>{col2_text}</h2>",
        unsafe_allow_html=True,
    )
    col3.subheader("**Performance**")
    col3_text = f"{portfolio_kpis['performance']:.0f} %"
    col3.markdown(
        f"<h2 style='color:{f'{RED}' if portfolio_kpis['performance'] < 0 else f'{GREEN}'};'>{col3_text}</h2>",
        unsafe_allow_html=True,
    )
    col4.subheader("**Volatility**")
    col4_text = f"{cur_vol * 100:.0f} %"
    col4.markdown(f"<h2 >{col4_text}</h2>", unsafe_allow_html=True)

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


def _asset_performance(
    portfolio_kpis,
    cur_vol,
    data_dict,
    portfolio_df,
    selected_benchmark_ticker,
    all_tickers_perf,
):
    selected_ticker = st.multiselect("Select Ticker:", tickers, default=tickers)
    ticker_data_dict = {ticker: all_tickers_perf[ticker] for ticker in selected_ticker}
    titles = [value["title"].iloc[0] for _, value in ticker_data_dict.items()]
    # st.write(data_dict['EVISO.MI'])

    final_data_dict = {**ticker_data_dict, **data_dict}
    final_titles = titles + selected_benchmark_ticker
    # st.write(final_titles)

    if selected_ticker:
        perf_chart = plot_performance(
            portfolio_df=None,
            items=final_data_dict,
            color=f"{ORANGE}",
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
    col0.subheader("**Asset**")
    col1.subheader("**Total Value**")
    col2.subheader("**Gain/Loss**")
    col3.subheader("**Performance**")
    col4.subheader("**Volatility**")
    for t in selected_ticker:
        df = ticker_data_dict[t].iloc[-1]
        vol = (ticker_data_dict[t]["Daily Return (%)"] / 100).std() * (252**0.5)

        col0.header(f"**{df['Ticker']}**")
        col1.header(f"€ {df['Market Value']:,.0f}")
        col2_text = f"€ {df['Unrealized Gains']:.2f}"
        col2.markdown(
            f"<h2 style='color:{f'{RED}' if df['Unrealized Gains'] < 0 else f'{GREEN}'};'>{col2_text}</h2>",
            unsafe_allow_html=True,
        )
        col3_text = f"{df['Performance (%)']:.0f} %"
        col3.markdown(
            f"<h2 style='color:{f'{RED}' if df['Performance (%)'] < 0 else f'{GREEN}'};'>{col3_text}</h2>",
            unsafe_allow_html=True,
        )
        col4_text = f"{vol * 100:.0f} %"
        col4.markdown(f"<h2 >{col4_text}</h2>", unsafe_allow_html=True)

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


if __name__ == "__main__":
    result = render_page() if render_page else None

    if result is not None:
        (
            my_portfolio,
            portfolio_df,
            all_tickers_perf,
            allocation_df,
            benchmarks,
            portfolio_kpis,
            tickers,
        ) = result

    render_performance_tab()

import streamlit as st

from portfolio_analysis.scripts.visualization.plots import (
    create_pie_chart,
    plot_asset_allocation_by_type,
)
from utils.streamlit_utils import _general_performance, render_page
from portfolio_analysis.scripts.constants.color import BLUE, GREEN, RED


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


def _home_allocation(allocation_df):
    st.write("**Allocation**")
    pills = st.pills(
        label=None, options=["Ticker", "Full Name", "Type"], default="Ticker"
    )

    if pills == "Ticker":
        # st.subheader("Allocation by Asset")
        pie_chart = create_pie_chart(allocation_df, x="Ticker")
        st.plotly_chart(pie_chart, use_container_width=True)
    if pills == "Full Name":
        # st.subheader("Allocation by Asset")
        pie_chart = create_pie_chart(allocation_df, x="Title")
        st.plotly_chart(pie_chart, use_container_width=True)
    elif pills == "Type":
        # st.subheader("Allocation by Asset")
        pie_chart = plot_asset_allocation_by_type(allocation_df)
        st.plotly_chart(pie_chart, use_container_width=True)


def _home_kpis_returns(portfolio_kpis):
    with st.container():
        with st.expander("Portfolio Returns"):
            col1, col2, col3 = st.columns(3)
            col1.write("**Daily Returns**")
            col1_text = f"€ {portfolio_kpis['returns']['daily']['abs']:.2f}"
            col1.markdown(
                f"<h2 style='color:{f'{RED}' if portfolio_kpis['returns']['daily']['abs'] < 0 else f'{GREEN}'};'>{col1_text}</h2>",
                unsafe_allow_html=True,
            )
            col1_text = f"{portfolio_kpis['returns']['daily']['pct']:.2f} %"
            col1.markdown(
                # f"<div style='text-align: center;'>"
                f"<div style='color:{f'{RED}' if portfolio_kpis['returns']['daily']['pct'] < 0 else f'{GREEN}'}; font-size:20px; font-weight:bold;'>{col1_text}</div>",
                unsafe_allow_html=True,
            )

            col2.write("**Weekly Returns**")
            col2_text = f"€ {portfolio_kpis['returns']['weekly']['abs']:.2f}"
            col2.markdown(
                f"<h2 style='color:{f'{RED}' if portfolio_kpis['returns']['daily']['abs'] < 0 else f'{GREEN}'};'>{col2_text}</h2>",
                unsafe_allow_html=True,
            )
            col2_text = f"{portfolio_kpis['returns']['weekly']['pct']:.2f} %"
            col2.markdown(
                # f"<div style='text-align: center;'>"
                f"<div style='color:{f'{RED}' if portfolio_kpis['returns']['weekly']['pct'] < 0 else f'{GREEN}'}; font-size:20px; font-weight:bold;'>{col2_text}</div>",
                unsafe_allow_html=True,
            )

            col3.write("**Monthly Returns**")
            col3_text = f"€ {portfolio_kpis['returns']['monthly']['abs']:.2f}"
            col3.markdown(
                f"<h2 style='color:{f'{RED}' if portfolio_kpis['returns']['daily']['abs'] < 0 else f'{GREEN}'};'>{col3_text}</h2>",
                unsafe_allow_html=True,
            )
            col3_text = f"{portfolio_kpis['returns']['monthly']['pct']:.2f} %"
            col3.markdown(
                # f"<div style='text-align: center;'>"
                f"<div style='color:{f'{RED}' if portfolio_kpis['returns']['monthly']['pct'] < 0 else f'{GREEN}'}; font-size:20px; font-weight:bold;'>{col3_text}</div>",
                unsafe_allow_html=True,
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
        with st.expander("Best/Worst Assets"):
            best_ticker = portfolio_kpis["best_ticker"]
            worst_ticker = portfolio_kpis["worst_ticker"]
            col1, col2, col3 = st.columns(3)

            col1.write("**Best Performing Ticker**")
            col1_text = f"{best_ticker['ticker']}"
            col1.markdown(f"<h3>{col1_text}</h3>", unsafe_allow_html=True)
            col2.write("**Value**")
            col2_text = f"€ {best_ticker['unrealized_gains']:.2f}"
            col2.markdown(f"<h3>{col2_text}</h3>", unsafe_allow_html=True)
            col3.write("**Performance**")
            col3_text = f"{best_ticker['performance']:.2f} %"
            col3.markdown(
                f"<h3 style='color:{f'{RED}' if best_ticker['performance'] < 0 else f'{GREEN}'};'>{col3_text}</h3>",
                unsafe_allow_html=True,
            )

            col1.divider()
            col2.divider()
            col3.divider()

            col1.write("**Worst Performing Ticker**")
            col1_text = f"{worst_ticker['ticker']}"
            col1.markdown(f"<h3>{col1_text}</h3>", unsafe_allow_html=True)
            col2.write("**Value**")
            col2_text = f"€ {worst_ticker['unrealized_gains']:.2f}"
            col2.markdown(f"<h3>{col2_text}</h3>", unsafe_allow_html=True)
            col3.write("**Performance**")
            col3_text = f"{worst_ticker['performance']:.2f} %"
            col3.markdown(
                f"<h3 style='color:{f'{RED}' if worst_ticker['performance'] < 0 else f'{GREEN}'};'>{col3_text}</h3>",
                unsafe_allow_html=True,
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

        render_home_tab(allocation_df, portfolio_kpis, portfolio_df)

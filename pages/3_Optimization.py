import pandas as pd
import streamlit as st

from portfolio_analysis.scripts.data.optimization import (
    SAME_RISK,
    STRATEGIES,
    TARGET_RETURN,
    optimize,
)
from portfolio_analysis.scripts.utils.utils import colorize
from portfolio_analysis.scripts.visualization.plots import plot_optimization
from utils.streamlit_utils import render_page


def render_optimization_tab(my_portfolio, allocation_df, portfolio_kpis, portfolio_df):
    if (
        my_portfolio is not None
        and allocation_df is not None
        and portfolio_kpis is not None
        and portfolio_df is not None
    ):
        st.header("Portfolio Optimization")

        target_return = None
        target_volatility = None
        risk_free_rate = 0.02
        curr_weights, cur_ret, cur_vol = my_portfolio.current_weights()
        curr_weights_sorted = (
            curr_weights["Allocation(%)"]
            .sort_values(ascending=False)
            .to_frame()
            .T.round(0)
        )
        _, mean_returns, cov_matrix = my_portfolio.get_returns_matrix()
        df_random, frontiers, port_opt = optimize(
            mean_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
            target_return=target_return,
            target_volatility=target_volatility,
        )

        cols = st.columns([1, 0.2, 1])
        with cols[0]:
            st.write("**Current Portfolio Allocation**")
            st.dataframe(curr_weights_sorted, use_container_width=True)

            fig = plot_optimization(df_random, frontiers, port_opt, (cur_ret, cur_vol))
            st.plotly_chart(fig, use_container_width=True)
        with cols[2]:
            pills = st.pills(
                "Choose your Strategy", options=STRATEGIES, default=STRATEGIES[0]
            )
            target_disabled = pills != TARGET_RETURN
            risk_disabled = pills != SAME_RISK

            input_cols = st.columns(2)
            with input_cols[0]:
                user_input_tr = st.number_input(
                    "Desired Annual Return (%)",
                    value=5.0,
                    step=1.0,
                    disabled=target_disabled,
                )
                target_return = user_input_tr / 100.0
            with input_cols[1]:
                user_input_tv = st.number_input(
                    "Desired Volatility (%)",
                    value=15.0,
                    step=1.0,
                    disabled=risk_disabled,
                )
                target_volatility = user_input_tv / 100.0

            df_random, frontiers, port_opt = optimize(
                mean_returns,
                cov_matrix,
                risk_free_rate=risk_free_rate,
                target_return=target_return,
                target_volatility=target_volatility,
            )
            st.divider()
            space = st.columns(3)
            space[0].metric(
                label="Expected Annual Return",
                value=f"{port_opt[pills]['ret'] * 100:.2f} %",
            )
            space[1].metric(
                label="Annual Volatility", value=f"{port_opt[pills]['vol'] * 100:.2f} %"
            )
            space[2].metric(
                label="Sharpe Ratio", value=f"{port_opt[pills]['sharpe']:.2f}"
            )
            st.divider()
        with cols[2]:
            st.write(f"**New Allocation** following **{pills} strategy**")
            opt_weights = pd.DataFrame(port_opt[pills]["weights"], index=["%"])
            opt_allocation = (opt_weights * 100).round(0)
            st.dataframe(opt_allocation, use_container_width=True)
            st.divider()

            st.write(f"**Allocation Difference from Current Portfolio**")
            diff_curr_weights = (
                curr_weights.rename(columns={"Allocation(%)": "%"})["%"].sort_index()
                / 100
            )
            # st.dataframe(diff_curr_weights.to_frame().T, use_container_width=True)
            diff_opt_weights = opt_weights.T.sort_index()
            # st.dataframe(diff_opt_weights.T, use_container_width=True)
            diff_weights = (diff_opt_weights.T - diff_curr_weights.T) * 100
            diff_weights = (
                diff_weights.style.applymap(colorize)
                .format("{:.0f}")
                .set_properties(**{"font-size": "24pt"})
            )

            st.dataframe(diff_weights, use_container_width=True)
            # st.text('Green coloured values -> ')


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

        render_optimization_tab(
            my_portfolio, allocation_df, portfolio_kpis, portfolio_df
        )

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from pypfopt.efficient_frontier import EfficientFrontier

MAX_SHARPE = "Max Sharpe"
MIN_VOLATILITY = "Min Volatility"
TARGET_RETURN = "Target Return"
SAME_RISK = "Same Risk"

STRATEGIES = [MAX_SHARPE, MIN_VOLATILITY, TARGET_RETURN, SAME_RISK]

def _random_portfolios_scatter(mean_returns, cov_matrix, risk_free_rate, n_portfolios=2000):
    n_assets = len(mean_returns)
    results = []
    for _ in range(n_portfolios):
        w = np.random.random(n_assets)
        w /= w.sum()
        ret = np.sum(w * mean_returns)
        vol = np.sqrt(w @ cov_matrix @ w)
        sharpe = (ret - risk_free_rate) / vol if vol != 0 else 0.0
        results.append([vol, ret, sharpe])
    df = pd.DataFrame(results, columns=["Volatility", "Return", "Sharpe"])
    return df



def _sample_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, n_points=50):
    """
    Sample the efficient frontier by scanning returns from min volatility to max Sharpe.
    """
    # Create separate instances for bounds
    ef_min = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1))
    ef_min.min_volatility()
    ret_minVol, _, _ = ef_min.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)

    ef_max = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1))
    ef_max.efficient_risk(target_volatility=0.2)
    ret_maxSharpe, _, _ = ef_max.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)

    ret_low = min(ret_minVol, ret_maxSharpe)
    ret_high = max(ret_minVol, ret_maxSharpe)
    test_returns = np.linspace(ret_low, ret_high, n_points)

    # Create a new instance for sampling
    ef_sample = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1))

    frontier_vols = []
    frontier_rets = []
    for tr in test_returns:
        try:
            ef_sample.efficient_return(target_return=tr)
            ret, vol, _ = ef_sample.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            frontier_vols.append(vol)
            frontier_rets.append(ret)
        except Exception:
            # Skip infeasible target returns
            pass

    return frontier_vols, frontier_rets


def optimize(mean_returns,
              cov_matrix,
              risk_free_rate,
              target_return=None,
              target_volatility=None,
              n_portfolios=2000):
    port_opt = {}

    # A) Random portfolio scatter
    df_random = _random_portfolios_scatter(mean_returns, cov_matrix, risk_free_rate, n_portfolios)

    # B) Create a fresh EfficientFrontier instance for the chosen strategy
    ef_strategy = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1))

    # MAX SHARPE
    ef_strategy.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef_strategy.clean_weights()
    chosen_ret, chosen_vol, chosen_sharpe = ef_strategy.portfolio_performance(risk_free_rate=risk_free_rate,
                                                                              verbose=False)
    port_opt[MAX_SHARPE] = {'ret': float(chosen_ret),
                            'vol': float(chosen_vol),
                            'sharpe': float(chosen_sharpe),
                            'weights': cleaned_weights}

    # MIN VOLATILITY
    ef_strategy = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1))
    ef_strategy.min_volatility()
    cleaned_weights = ef_strategy.clean_weights()
    chosen_ret, chosen_vol, chosen_sharpe = ef_strategy.portfolio_performance(risk_free_rate=risk_free_rate,
                                                                              verbose=False)
    port_opt[MIN_VOLATILITY] = {'ret': float(chosen_ret),
                                'vol': float(chosen_vol),
                                'sharpe': float(chosen_sharpe),
                                'weights': cleaned_weights}

    # TARGET
    if target_return is None:
        target_return = 0.1
    ef_strategy = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1))
    ef_strategy.efficient_return(target_return=target_return)
    cleaned_weights = ef_strategy.clean_weights()
    chosen_ret, chosen_vol, chosen_sharpe = ef_strategy.portfolio_performance(risk_free_rate=risk_free_rate,
                                                                              verbose=False)
    port_opt[TARGET_RETURN] = {'ret': float(chosen_ret),
                               'vol': float(chosen_vol),
                               'sharpe': float(chosen_sharpe),
                               'weights': cleaned_weights}

    if target_volatility is None:
        target_volatility = 0.1
    ef_strategy = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1))
    ef_strategy.efficient_risk(target_volatility=target_volatility)
    cleaned_weights = ef_strategy.clean_weights()
    chosen_ret, chosen_vol, chosen_sharpe = ef_strategy.portfolio_performance(risk_free_rate=risk_free_rate,
                                                                              verbose=False)
    port_opt[SAME_RISK] = {'ret': float(chosen_ret),
                           'vol': float(chosen_vol),
                           'sharpe': float(chosen_sharpe),
                           'weights': cleaned_weights}

    frontier_vols, frontier_rets = _sample_efficient_frontier(mean_returns,
                                                              cov_matrix,
                                                              risk_free_rate,
                                                              n_points=50)
    frontiers = {'vols': frontier_vols,
                 'rets': frontier_rets}

    return df_random, frontiers, port_opt
def optimize_and_plot(mean_returns,
                      cov_matrix,
                      risk_free_rate,
                      current_portfolio_ret_vol=None,
                      target_return=None,
                      target_volatility=None,
                      n_portfolios=3000):
    port_opt = {}

    # A) Random portfolio scatter
    df_random = _random_portfolios_scatter(mean_returns, cov_matrix, risk_free_rate, n_portfolios)

    # B) Create a fresh EfficientFrontier instance for the chosen strategy
    ef_strategy = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1))

    # MAX SHARPE
    ef_strategy.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef_strategy.clean_weights()
    chosen_ret, chosen_vol, chosen_sharpe = ef_strategy.portfolio_performance(risk_free_rate=risk_free_rate,
                                                                              verbose=False)
    port_opt[MAX_SHARPE] = {'ret': float(chosen_ret),
                               'vol': float(chosen_vol),
                               'sharpe': float(chosen_sharpe),
                               'weights': cleaned_weights}

    # MIN VOLATILITY
    ef_strategy = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1))
    ef_strategy.min_volatility()
    cleaned_weights = ef_strategy.clean_weights()
    chosen_ret, chosen_vol, chosen_sharpe = ef_strategy.portfolio_performance(risk_free_rate=risk_free_rate,
                                                                              verbose=False)
    port_opt[MIN_VOLATILITY] = {'ret': float(chosen_ret),
                               'vol': float(chosen_vol),
                               'sharpe': float(chosen_sharpe),
                               'weights': cleaned_weights}

    # TARGET
    if target_return is not None:
        ef_strategy = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1))
        ef_strategy.efficient_return(target_return=target_return)
        cleaned_weights = ef_strategy.clean_weights()
        chosen_ret, chosen_vol, chosen_sharpe = ef_strategy.portfolio_performance(risk_free_rate=risk_free_rate,
                                                                                  verbose=False)
        port_opt[TARGET_RETURN] = {'ret': float(chosen_ret),
                               'vol': float(chosen_vol),
                               'sharpe': float(chosen_sharpe),
                               'weights': cleaned_weights}

    if target_return is not None:
        ef_strategy = EfficientFrontier(mean_returns, cov_matrix, weight_bounds=(0, 1))
        ef_strategy.efficient_risk(target_volatility=target_volatility)
        cleaned_weights = ef_strategy.clean_weights()
        chosen_ret, chosen_vol, chosen_sharpe = ef_strategy.portfolio_performance(risk_free_rate=risk_free_rate,
                                                                                  verbose=False)
        port_opt[SAME_RISK] = {'ret': float(chosen_ret),
                               'vol': float(chosen_vol),
                               'sharpe': float(chosen_sharpe),
                               'weights': cleaned_weights}

    frontier_vols, frontier_rets = _sample_efficient_frontier(mean_returns,
                                                              cov_matrix,
                                                              risk_free_rate,
                                                              n_points=50)

    # F) Build the Plotly figure
    fig = px.scatter(
        df_random,
        x="Volatility",
        y="Return",
        color="Sharpe",
        color_continuous_scale="RdBu",
        title=f"Efficient Frontier"
    )
    fig.update_traces(marker=dict(size=4))

    fig.add_trace(go.Scatter(
        x=frontier_vols,
        y=frontier_rets,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='black', width=2)
    ))

    for strategy, color in zip(port_opt.keys(), ['orange', 'brown', 'purple', 'pink']):
        fig.add_trace(go.Scatter(
            x=[port_opt[strategy]['vol']],
            y=[port_opt[strategy]['ret']],
            mode='markers',
            marker=dict(color=color, size=12, symbol='star'),
            name=strategy,
        ))

    if current_portfolio_ret_vol is not None:
        cur_ret, cur_vol = current_portfolio_ret_vol
        fig.add_trace(go.Scatter(
            x=[cur_vol],
            y=[cur_ret],
            mode='markers',
            marker=dict(color='red', size=12, symbol='diamond'),
            name='Current Portfolio'
        ))

    fig.update_layout(
        xaxis_title="Annualized Volatility (σ)",
        yaxis_title="Annualized Return (μ)",
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.1,  # slightly above the plot
            xanchor="center"
        )
    )

    return fig, port_opt


def render_optimization_tab(mean_returns,
                            cov_matrix,
                            current_portfolio_ret_vol=None):
    """
    A Streamlit UI that:
    - Lets user pick the strategy
    - Optionally enters target return or volatility
    - Plots the chart with the frontier & random portfolios
    """
    st.header("Portfolio Optimization")

    # Let user pick strategy
    strategy = st.selectbox(
        "Select Strategy",
        ["Max Sharpe", "Min Volatility", "Target Return", "Same Risk"]
    )
    risk_free_rate = st.number_input("Risk-Free Rate", value=0.02)
    target_return = None
    target_volatility = None

    if strategy == "Target Return":
        user_input_tr = st.number_input("Desired Annual Return (%)", value=10.0, step=1.0)
        target_return = user_input_tr / 100.0
    elif strategy == "Same Risk":
        user_input_tv = st.number_input("Desired Volatility (%)", value=15.0, step=1.0)
        target_volatility = user_input_tv / 100.0

    if st.button("Optimize"):
        fig, portfolio_info = optimize_and_plot(
            mean_returns,
            cov_matrix,
            strategy=strategy,
            risk_free_rate=risk_free_rate,
            current_portfolio_ret_vol=current_portfolio_ret_vol,
            target_return=target_return,
            target_volatility=target_volatility
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Chosen Portfolio")
        st.write(portfolio_info)


# ---------------------------------------------------------------------
# EXAMPLE USAGE in your main Streamlit script
# ---------------------------------------------------------------------

def main():
    st.title("Portfolio Analysis with 'Current Portfolio' & Smoother Frontier")

    # 1) Suppose we have:
    # mean_returns, cov_matrix from your daily returns
    # We'll just create dummy data here for illustration:
    tickers = ["AAPL", "GOOG", "TSLA", "AMZN"]
    mean_returns = pd.Series([0.15, 0.12, 0.2, 0.10], index=tickers)  # e.g. 15%, 12%, 20%, 10%
    cov_matrix = pd.DataFrame([
        [0.04, 0.01, 0.02, 0.01],
        [0.01, 0.03, 0.015, 0.005],
        [0.02, 0.015, 0.05, 0.02],
        [0.01, 0.005, 0.02, 0.02]
    ], index=tickers, columns=tickers)

    # 2) "Current portfolio" - e.g. we have 50% AAPL, 25% GOOG, 15% TSLA, 10% AMZN
    # We can compute the annualized return & volatility:
    current_weights = np.array([0.5, 0.25, 0.15, 0.1])
    cur_ret = np.sum(current_weights * mean_returns)
    cur_vol = np.sqrt(current_weights @ cov_matrix.values @ current_weights.T)
    current_portfolio_ret_vol = (cur_ret, cur_vol)

    st.write("**Current portfolio**: 50% AAPL, 25% GOOG, 15% TSLA, 10% AMZN.")
    st.write(f"- Return: {cur_ret * 100:.2f}%")
    st.write(f"- Volatility: {cur_vol * 100:.2f}%")

    # 3) Show the optimization tab
    render_optimization_tab(mean_returns, cov_matrix, current_portfolio_ret_vol)


if __name__ == "__main__":
    main()

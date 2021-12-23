import pandas as pd


def annualized_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compunded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compunded_growth ** (periods_per_year / n_periods) - 1


def annualized_vol(r, periods_per_year):
    """
    Annualises vol for a set of returns
    We should infer the periods per year
    """
    return r.std() * (periods_per_year ** 0.5)


def sharpe_ratio(r, risk_free_rate, periods_per_year):
    """
    Computes the annualised sharpe ratio of a set of returns
    """
    # convert the annual risk free rate to per period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = r - rf_per_period
    annual_rets = annualized_rets(excess_returns, periods_per_year)
    annual_vol = annualized_vol(r, periods_per_year)
    return annual_rets / annual_vol


def portfolio_return(weights, returns):
    """
    Weights -> Returns
    @ is matrix multiplication
    """
    return weights.T @ returns


def portfolio_vol(weights, comvat):
    """
    Weights -> Volatility
    @ is matrix multiplication
    """
    return (weights.T @ comvat @ weights) ** 0.5  # weights.T @ comvat @ weights gives us variance


def max_drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns
    Computes a dataframe that contains
    wealth index
    previous peaks
    drawdowns
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdowns": drawdowns
    })


def get_er(returns_df, periods_per_year):
    er_values = []
    columns = []
    for col in returns_df.columns:
        returns = returns_df[col].dropna()
        if len(returns) > 1:
            er_values.append(annualized_rets(returns, periods_per_year))
            columns.append(col)
    er = pd.Series(er_values, index=columns)

    # er = annualized_rets(returns_df, periods_per_year)
    # er = er[returns_df.columns]

    return er


def get_cov(returns_df):
    cov = returns_df.dropna().cov()
    cov = cov.loc[returns_df.columns, returns_df.columns]

    return cov


def get_df_from_dict(data_dict):
    result_df = pd.DataFrame()
    for x in data_dict:
        if len(data_dict[x]) > 1:
            result_df = pd.concat((result_df, data_dict[x].to_frame(x)), axis=1)

    return result_df

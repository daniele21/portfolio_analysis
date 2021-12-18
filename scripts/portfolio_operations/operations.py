import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm


def semi_deviation(r):
    """
    Returns the semi deviation, aka negative semi deviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def skewness(r):
    """
    Alternate to scipy.stats.skew()
    Computes the skewness of the supplied Series or  DataFrame
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()
    return exp / sigma_r ** 3


def kurtosis(r):
    """
    Alternate to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()
    return exp / sigma_r ** 4


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not.
    Test is applied at 1% level by default
    Returns True if the hypothesis of normality is accepted, else False.
    """
    statistic, pvalue = scipy.stats.jarque_bera(r)
    return pvalue > level


def var_historic(r, levels=5):
    """
    Returns the historic value at risk at a specified level
    i.e returns the level such that "level" percent of returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, levels=levels)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, levels, axis=0)
    else:
        raise TypeError("Series or DataFrame expected")


def var_gaussian(r, levels=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If modified is  True, then the modified VaR is returned
    using Cornish-Fisher modifications
    """
    # compute the Z score assuming the distribution is Gaussian
    z = norm.ppf(levels / 100)

    if modified:
        # calculate the Z score based on kurtosis and skewness
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z ** 2 + 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 +
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )

    return -(r.mean() + z * r.std(ddof=0))


def cvar_historic(r, levels=5):
    """
    Computes the conditional VaR or CVaR of the Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, levels=levels)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, levels=levels)
    else:
        raise TypeError("Expected the input to be DataFrame or Series")


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


def get_cov(returns_dict):
    returns_df = pd.DataFrame()
    for x in returns_dict:
        if len(returns_dict[x]) > 1:
            returns_df = pd.concat((returns_df, returns_dict[x].to_frame(x)), axis=1)

    cov = returns_df.dropna().cov()
    cov = cov.loc[returns_df.columns, returns_df.columns]

    return cov

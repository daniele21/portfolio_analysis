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

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.operations.time_series import portfolio_return, portfolio_vol


def minimize_vol(target_return, er, cov):
    """
    target_return -> weight
    """
    n = er.shape[0]
    init = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    # constraints for the weights
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
        # the target_return should be the one obtained from portfolio
    }

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(
        portfolio_vol, init,
        args=(cov,), method="SLSQP",
        options={'disp': False},
        constraints=(return_is_target, weights_sum_to_1),
        bounds=bounds
    )
    return results.x


def optimal_weights(n_points, er, cov,
                    min_target: float = None,
                    max_target: float = None):
    """
    list  of returns to run the optimizer on to get the weights
    """
    min_target = er.min() if min_target is None else min_target
    max_target = er.max() if max_target is None else max_target

    # target_rs = np.linspace(er.min(), er.max(), n_points)
    target_rs = np.linspace(min_target, max_target, n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def get_efficient_frontier(n_points: int, er, cov):
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "returns": rets,
        "volatility": vols,
        "weights": weights
    })

    ticker_weights = np.array(weights)
    ticker_weights_df = pd.DataFrame(ticker_weights, columns=cov.columns)

    ef = pd.concat((ef, ticker_weights_df), axis=1)
    ef = ef.set_index('volatility')

    return ef


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }

    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

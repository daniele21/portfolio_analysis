import numpy as np
import pandas as pd


def pandas_series_to_float32(series: pd.Series):
    return series.astype(np.float32)

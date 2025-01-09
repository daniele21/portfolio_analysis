import pandas as pd
import yfinance as yf
from datetime import date


def fetch_benchmark_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, group_by="ticker")

    if df.empty:
        raise ValueError(f"No data returned for benchmark {symbol}.")

    # If columns are a MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() for col in df.columns.values]
        # Now columns might look like '^GSPC_Open', '^GSPC_Close', etc.

        # Rename them so that we have a "Close" column
        rename_dict = {}
        for col in df.columns:
            lc = col.lower()
            if lc.endswith("_open"):
                rename_dict[col] = "Open"
            elif lc.endswith("_high"):
                rename_dict[col] = "High"
            elif lc.endswith("_low"):
                rename_dict[col] = "Low"
            elif lc.endswith("_close"):
                rename_dict[col] = "Close"
            elif lc.endswith("_volume"):
                rename_dict[col] = "Volume"
        df.rename(columns=rename_dict, inplace=True)
    else:
        # Single-level columns case:
        df.columns = [col.title() for col in df.columns]

    # Now you can safely dropna on "Close"
    df = df.dropna(subset=["Close"]).copy()

    # Sort by date index
    df.sort_index(inplace=True)

    # Compute daily returns in percent
    df["Daily Bench Return (%)"] = df["Close"].pct_change() * 100

    # Convert the index to a column named 'Date'
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "Date"}, inplace=True)

    return df


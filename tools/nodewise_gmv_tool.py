
"""GMV portfolio construction and evaluation on DJ30 data via nodewise regression."""

from typing import Optional, Union, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC


def load_dj30_excess_returns(
    csv_path: str = "data/dj30_data_full.csv",
    price_prefix: str = "close_",
    rf_col: str = "TB3MS",
    trading_days: int = 252,
    date_col: str = "Date",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load DJ30 prices and 3M T-bill, return excess returns in a date window.

    If `tickers` is given, restrict to that subset (e.g., top-10 names).

    Note: This function handles the long-format DJ30 data (with 'ticker' column)
    and transforms it to wide format, then merges with TB3MS from macro factors.
    """

    # Load DJ30 data (long format with ticker column)
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataset.")
    if 'ticker' not in df.columns:
        raise ValueError("'ticker' column not found in DJ30 dataset.")

    df[date_col] = pd.to_datetime(df[date_col])

    # Filter by date range before pivoting
    if start_date is not None:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.to_datetime(end_date)]
    if df.empty:
        raise ValueError("No data in the specified date range.")

    # Filter by tickers if specified
    if tickers is not None:
        tickers_upper = [t.upper() for t in tickers]
        df = df[df['ticker'].str.upper().isin(tickers_upper)]
        if df.empty:
            raise ValueError(f"None of the requested tickers found in data: {tickers}")

    # Pivot from long to wide format: Date x Ticker
    # Use adj_close if available, otherwise close
    price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
    df_wide = df.pivot(index=date_col, columns='ticker', values=price_col)

    # Drop any rows with all NaN
    df_wide = df_wide.dropna(how='all')

    if df_wide.empty:
        raise ValueError("No price data available after pivoting.")

    # Rename columns to match expected format (close_AAPL, close_MSFT, etc.)
    df_wide.columns = [f"{price_prefix}{col}" for col in df_wide.columns]

    # Load macro factors to get TB3MS (risk-free rate)
    import os
    macro_path = os.path.join(os.path.dirname(csv_path), "macro_factors_new.csv")
    if not os.path.exists(macro_path):
        raise ValueError(f"Macro factors file not found at {macro_path}. Cannot load {rf_col}.")

    macro_df = pd.read_csv(macro_path)
    macro_df[date_col] = pd.to_datetime(macro_df[date_col])
    macro_df = macro_df.set_index(date_col)

    if rf_col not in macro_df.columns:
        raise ValueError(f"Risk-free column '{rf_col}' not found in macro factors dataset.")

    # Merge price data with risk-free rate
    df_merged = df_wide.join(macro_df[[rf_col]], how='inner')

    if df_merged.empty:
        raise ValueError("No overlapping dates between DJ30 prices and macro factors.")

    df_merged = df_merged.sort_index()

    # Get price columns
    price_cols = [c for c in df_merged.columns if c.startswith(price_prefix)]
    if not price_cols:
        raise ValueError(f"No columns starting with '{price_prefix}' found after pivot.")

    # Calculate returns
    prices = df_merged[price_cols].apply(pd.to_numeric, errors="coerce")
    rets = prices.pct_change().iloc[1:]

    # Calculate daily risk-free rate from annual percentage
    rf_ann = pd.to_numeric(df_merged[rf_col], errors="coerce")
    rf_daily = (rf_ann / 100.0) / trading_days
    rf_daily = rf_daily.iloc[1:]  # align with returns

    # Calculate excess returns
    excess = rets.sub(rf_daily.values, axis=0)
    excess = excess.dropna(how='any')
    if excess.empty:
        raise ValueError("No usable excess-return rows after dropping NaNs.")

    # Remove 'close_' prefix from column names: 'close_AAPL' -> 'AAPL'
    cols_tickers = [c[len(price_prefix):] for c in price_cols]
    excess.columns = cols_tickers

    return excess


def estimate_precision_nodewise(
    excess_returns: pd.DataFrame,
    criterion: str = "bic",
) -> pd.DataFrame:
    """Naive nodewise Lasso precision estimator (Meinshausen–Bühlmann style)."""

    X = excess_returns.values
    X = X - X.mean(axis=0, keepdims=True)  # de-mean per asset
    T, p = X.shape
    tickers = list(excess_returns.columns)

    Theta = np.zeros((p, p), dtype=float)

    for j in range(p):
        y = X[:, j]
        mask = np.ones(p, dtype=bool)
        mask[j] = False
        X_j = X[:, mask]

        model = LassoLarsIC(criterion=criterion)
        model.fit(X_j, y)
        gamma_hat = model.coef_

        resid = y - X_j @ gamma_hat
        tau2 = float(np.mean(resid**2))
        if tau2 <= 0 or not np.isfinite(tau2):
            tau2 = 1e-6
        theta_jj = 1.0 / tau2

        gamma_full = np.zeros(p)
        gamma_full[mask] = gamma_hat

        # Θ_j,-j = -Θ_jj γ_j, Θ_jj on diagonal
        Theta[j, :] = -theta_jj * gamma_full
        Theta[j, j] = theta_jj

    # enforce symmetry
    Theta = 0.5 * (Theta + Theta.T)

    return pd.DataFrame(Theta, index=tickers, columns=tickers)


def compute_gmv_weights(precision: pd.DataFrame) -> pd.Series:
    """GMV weights: w ∝ Θ 1, normalized to sum to one."""

    Theta = precision.values
    p = Theta.shape[0]
    ones = np.ones(p)

    num = Theta @ ones
    denom = float(ones @ num)
    if denom == 0 or not np.isfinite(denom):
        raise ValueError("Denominator in GMV formula is zero or non-finite.")

    w = num / denom
    return pd.Series(w, index=precision.index, name="GMV_weight")


def nodewise_gmv_from_csv(
    csv_path: str = "data/dj30_data_full.csv",
    price_prefix: str = "close_",
    rf_col: str = "TB3MS",
    trading_days: int = 252,
    criterion: str = "bic",
    date_col: str = "Date",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tickers: Optional[List[str]] = None,
) -> pd.Series:
    """Convenience wrapper: CSV -> excess returns -> precision -> GMV weights."""

    excess = load_dj30_excess_returns(
        csv_path=csv_path,
        price_prefix=price_prefix,
        rf_col=rf_col,
        trading_days=trading_days,
        date_col=date_col,
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
    )
    precision = estimate_precision_nodewise(excess, criterion=criterion)
    weights = compute_gmv_weights(precision)
    return weights


def evaluate_portfolio_from_weights(
    csv_path: str = "data/dj30_data_full.csv",
    weights: Union[pd.Series, Dict[str, float]] = None,
    price_prefix: str = "close_",
    rf_col: str = "TB3MS",
    trading_days: int = 252,
    date_col: str = "Date",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, object]:
    """Given weights, compute realized excess-return stats on a holdout window.

    Returns a dict with daily and annualized mean/variance/std and Sharpe.
    """

    if isinstance(weights, dict):
        w = pd.Series(weights, dtype=float)
    else:
        w = weights.astype(float)

    # load excess returns in the evaluation window
    excess = load_dj30_excess_returns(
        csv_path=csv_path,
        price_prefix=price_prefix,
        rf_col=rf_col,
        trading_days=trading_days,
        date_col=date_col,
        start_date=start_date,
        end_date=end_date,
    )

    # align tickers: intersection of weights and available columns
    common = sorted(set(excess.columns).intersection(w.index))
    if not common:
        raise ValueError("No overlapping tickers between weights and excess returns.")

    excess = excess[common]
    w = w[common]

    # renormalize to sum to one over the common names
    total = float(w.sum())
    if total == 0 or not np.isfinite(total):
        raise ValueError("Weights sum to zero or non-finite after alignment.")
    w = w / total

    # portfolio excess return series
    port_excess = excess.mul(w, axis=1).sum(axis=1)

    if port_excess.empty:
        raise ValueError("No non-missing portfolio returns in the evaluation window.")

    mean_d = float(port_excess.mean())
    var_d = float(port_excess.var(ddof=1))
    std_d = float(port_excess.std(ddof=1))
    sharpe_d = float(mean_d / std_d) if std_d > 0 else float("nan")

    ann = trading_days
    mean_a = mean_d * ann
    var_a = var_d * ann
    std_a = std_d * np.sqrt(ann)
    sharpe_a = sharpe_d * np.sqrt(ann) if std_d > 0 else float("nan")

    out = {
        "n_obs": int(len(port_excess)),
        "mean_daily": mean_d,
        "var_daily": var_d,
        "std_daily": std_d,
        "sharpe_daily": sharpe_d,
        "mean_annualized": mean_a,
        "var_annualized": var_a,
        "std_annualized": std_a,
        "sharpe_annualized": sharpe_a,
    }
    return out

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

def summary_stats(df: pd.DataFrame) -> dict:
    """
    Lightweight preflight diagnostics. Paste into: src/eda/forecastability.py
    """
    n_days = len(df)
    missing = int(df["y"].isna().sum())
    missing_pct = 100 * missing / max(1, n_days)
    span_days = int((df.index.max() - df.index.min()).days) + 1
    zero_frac = float((df["y"] == 0).mean())
    return {
        "n_days": n_days,
        "span_days": span_days,
        "missing_days": missing,
        "missing_pct": missing_pct,
        "zero_fraction": zero_frac,
        "intermittent": zero_frac >= 0.4,
    }

def seasonal_strength(series: pd.Series, nlags: int = 365) -> dict:
    """
    Quick seasonality proxies using ACF at weekly and annual lags.
    Paste into: src/eda/forecastability.py
    """
    s = series.dropna().astype(float)
    if len(s) < 14:
        return {"acf7": float("nan"), "acf365": float("nan")}
    a = acf(s, nlags=min(nlags, len(s) - 1), fft=True)
    return {"acf7": float(a[7]) if len(a) > 7 else float("nan"),
            "acf365": float(a[365]) if len(a) > 365 else float("nan")}
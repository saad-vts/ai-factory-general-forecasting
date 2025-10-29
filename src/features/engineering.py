import numpy as np
import pandas as pd

def make_calendar_features(index: pd.DatetimeIndex, weekend_days=[5,6]) -> pd.DataFrame:
    """
    Calendar/exogenous features used by most models.
    Paste into: src/features/engineering.py
    """
    cal = pd.DataFrame(index=index)
    cal["dow"] = cal.index.weekday
    cal["is_weekend"] = cal["dow"].isin(weekend_days).astype(int)
    cal["month"] = cal.index.month
    cal["dayofyear"] = cal.index.dayofyear
    cal["sin_annual"] = np.sin(2 * np.pi * cal["dayofyear"] / 365.25)
    cal["cos_annual"] = np.cos(2 * np.pi * cal["dayofyear"] / 365.25)
    return cal

def make_lags_rolls(y: pd.Series, lags=(1,7,14), roll_windows=(7,)) -> pd.DataFrame:
    """
    Create lag and rolling-stat features (past-only).
    Paste into: src/features/engineering.py
    """
    feat = pd.DataFrame(index=y.index)
    for L in lags:
        feat[f"lag_{L}"] = y.shift(L)
    for w in roll_windows:
        feat[f"roll{w}_mean"] = y.rolling(w).mean()
        feat[f"roll{w}_std"]  = y.rolling(w).std()
    return feat
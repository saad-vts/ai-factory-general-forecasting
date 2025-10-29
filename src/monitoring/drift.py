import numpy as np
import pandas as pd

def psi_score(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between arrays a (baseline) and b (current).
    Paste into: src/monitoring/drift.py
    """
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    qs = np.quantile(a, np.linspace(0,1,bins+1))
    qs[0], qs[-1] = -np.inf, np.inf
    a_bins = np.histogram(a, bins=qs)[0] / max(1, len(a))
    b_bins = np.histogram(b, bins=qs)[0] / max(1, len(b))
    return float(np.sum((b_bins - a_bins) * np.log((b_bins + 1e-9)/(a_bins + 1e-9))))

def rolling_mad_cp(y: pd.Series, window: int = 30, z_thresh: float = 4.0):
    """
    Change-point flagging using rolling MAD z-scores.
    Paste into: src/monitoring/drift.py
    """
    med = y.rolling(window, min_periods=max(5, window//2)).median()
    mad = (y - med).abs().rolling(window, min_periods=max(5, window//2)).median() * 1.4826
    z = (y - med) / mad.replace(0, np.nan)
    flag = (z.abs() > z_thresh).fillna(False)
    return z, flag
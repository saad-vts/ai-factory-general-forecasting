import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_leakage(features: pd.DataFrame, lookback_rows: int = 40) -> list:
    """
    Heuristic: lag/roll features should show NaNs in the earliest rows.
    Paste into: src/features/diagnostics.py
    """
    suspect = []
    for c in features.columns:
        if ("roll" in c or c.startswith("lag_")):
            if features[c].iloc[:lookback_rows].isna().sum() == 0:
                suspect.append(c)
    return suspect

def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """
    Return VIF table for numeric predictors. Paste into: src/features/diagnostics.py
    """
    Xn = X.select_dtypes(include=[np.number]).dropna()
    if Xn.shape[0] <= Xn.shape[1] + 1:
        return pd.DataFrame(columns=["feature","vif"])
    Xc = sm.add_constant(Xn)
    rows = []
    for i, col in enumerate(Xn.columns):
        try:
            v = variance_inflation_factor(Xc.values, i+1)
        except Exception:
            v = np.nan
        rows.append({"feature": col, "vif": float(v)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)
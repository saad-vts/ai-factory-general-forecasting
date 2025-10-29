import numpy as np
import pandas as pd

def corr_screen(endog: pd.Series, cand_exog: pd.DataFrame, min_abs_corr: float = 0.10) -> list:
    """
    Keep candidate exog with |corr| >= min_abs_corr. Paste into: src/selection/feature_selector.py
    """
    if cand_exog is None or cand_exog.empty:
        return []
    corr = cand_exog.apply(lambda s: float(np.corrcoef(s.reindex(endog.index).fillna(0), endog.fillna(0))[0,1]))
    return corr.index[corr.abs() >= min_abs_corr].tolist()

def select_external_features(endog: pd.Series,
                             base_exog: pd.DataFrame,
                             cand_exog: pd.DataFrame,
                             uplift_pp: float = 1.0,
                             rolling_backtest_fn = None):
    """
    Pipeline: corr -> optional uplift test (rolling_backtest_fn returns list of MPES)
    Paste into: src/selection/feature_selector.py
    """
    keep = corr_screen(endog, cand_exog)
    if not keep:
        return base_exog, []
    Xcand = cand_exog[keep].copy()
    if rolling_backtest_fn is None:
        # no budgeted test available, keep screened features
        return base_exog.join(Xcand), keep
    mpes_base = rolling_backtest_fn(endog, base_exog)
    mpes_plus = rolling_backtest_fn(endog, base_exog.join(Xcand))
    med_base = np.median(mpes_base) if len(mpes_base) else np.inf
    med_plus = np.median(mpes_plus) if len(mpes_plus) else np.inf
    uplift = med_base - med_plus
    if uplift >= uplift_pp:
        return base_exog.join(Xcand), keep
    return base_exog, []
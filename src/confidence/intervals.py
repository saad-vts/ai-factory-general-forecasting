import numpy as np
import pandas as pd

def conformal_intervals_from_residuals(y_true, y_pred, y_pred_future, alpha=0.1):
    """
    Generate conformal prediction intervals based on validation residuals.
    
    Args:
        y_true: Actual values (validation set)
        y_pred: Predicted values (validation set)
        y_pred_future: Future predictions (can be Series or array)
        alpha: Significance level (default 0.1 for 90% intervals)
    
    Returns:
        DataFrame or dict with 'lower' and 'upper' bounds
    """
    # Compute residuals
    residuals = np.abs(y_true - y_pred)
    
    # Compute quantile for conformal intervals
    q = np.quantile(residuals, 1 - alpha)
    
    # Convert future predictions to array if it's a Series
    if isinstance(y_pred_future, pd.Series):
        future_values = y_pred_future.values
        future_index = y_pred_future.index
    else:
        future_values = y_pred_future
        future_index = None
    
    # Generate intervals
    lower = future_values - q
    upper = future_values + q
    
    # Return with index if available
    if future_index is not None:
        return pd.DataFrame({"lower": lower, "upper": upper}, index=future_index)
    else:
        return {"lower": lower, "upper": upper}

def bootstrap_ci(yhat: pd.Series, residuals: np.ndarray, draws: int = 500, alpha: float = 0.05, seed: int = 42):
    """
    Bootstrap residuals to form empirical CIs. Paste into: src/confidence/intervals.py
    """
    if residuals is None or len(residuals) < 20:
        return None
    rng = np.random.default_rng(seed)
    sims = np.zeros((draws, len(yhat)))
    for i in range(draws):
        e = rng.choice(residuals, size=len(yhat), replace=True)
        sims[i, :] = yhat.values + e
    lo = np.quantile(sims, alpha/2, axis=0)
    hi = np.quantile(sims, 1 - alpha/2, axis=0)
    return pd.DataFrame({"lower": lo, "upper": hi}, index=yhat.index)
import time
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarimax(y: pd.Series, exog: pd.DataFrame = None, order=(1,1,1), seasonal_order=(1,1,1,7), budget_sec: int = 90):
    """
    Fit SARIMAX with a simple time budget guard. Returns result object or None on fail.
    Paste into: src/models/sarimax.py
    """
    try:
        t0 = time.time()
        m = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
        res = m.fit(disp=False)
        if time.time() - t0 > budget_sec:
            raise TimeoutError("SARIMAX fit exceeded budget")
        return res
    except Exception:
        return None

def forecast_sarimax(res, steps: int, exog_future: pd.DataFrame = None):
    if res is None:
        return None, None
    fc = res.get_forecast(steps=steps, exog=exog_future)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    ci.columns = ["lower", "upper"]
    return mean, ci
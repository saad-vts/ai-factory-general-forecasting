import pandas as pd
import numpy as np

def fetch_dummy_weather(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Example external signal generator (dummy). Replace with real API integration.
    Paste into: src/external/exog_fetchers.py
    """
    rng = np.random.default_rng(0)
    temp = 25 + 5 * np.sin(2 * np.pi * index.dayofyear / 365.25) + rng.normal(0, 1, len(index))
    return pd.DataFrame({"temp_c": temp}, index=index)
import pandas as pd

def make_country_holidays(index: pd.DatetimeIndex, country_code: str = "AE") -> pd.DataFrame:
    """
    Create DataFrame with is_holiday binary column for given index. Paste into: src/external/holidays.py
    Falls back to zeros if 'holidays' package not available.
    """
    try:
        import holidays as pyholidays
        cal = pyholidays.country_holidays(country_code)
        ser = pd.Series([1 if d.date() in cal else 0 for d in index], index=index)
        return pd.DataFrame({"is_holiday": ser})
    except Exception:
        return pd.DataFrame({"is_holiday": pd.Series(0, index=index)})
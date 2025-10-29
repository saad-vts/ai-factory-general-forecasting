import pytest
import pandas as pd
from src.features.engineering import make_calendar_features, make_lags_rolls
from src.features.diagnostics import check_leakage, vif_table

def test_calendar_features(sample_df):
    cal = make_calendar_features(sample_df.index)
    assert isinstance(cal, pd.DataFrame)
    assert 'is_weekend' in cal.columns
    assert 'sin_annual' in cal.columns
    assert len(cal) == len(sample_df)

def test_lags_rolls(sample_ts):
    feat = make_lags_rolls(sample_ts, lags=[1,7])
    assert isinstance(feat, pd.DataFrame)
    assert 'lag_1' in feat.columns
    assert 'lag_7' in feat.columns
    assert feat.iloc[:7].isna().any().all()  # first 7 rows should have NaNs

def test_leakage_check(sample_ts):
    feat = make_lags_rolls(sample_ts)
    leaks = check_leakage(feat)
    assert isinstance(leaks, list)
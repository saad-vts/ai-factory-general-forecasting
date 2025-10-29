import pytest
import pandas as pd
import numpy as np
from src.eda.forecastability import summary_stats, seasonal_strength
from src.eda.summary import get_basic_stats, get_temporal_stats

def test_summary_stats(sample_df):
    stats = summary_stats(sample_df)
    assert isinstance(stats, dict)
    assert 'n_days' in stats
    assert 'missing_pct' in stats
    assert stats['n_days'] == len(sample_df)

def test_seasonal_strength(sample_ts):
    strength = seasonal_strength(sample_ts)
    assert isinstance(strength, dict)
    assert 'acf7' in strength
    assert 'acf365' in strength
    assert not np.isnan(strength['acf7'])

def test_basic_stats(sample_ts):
    stats = get_basic_stats(sample_ts)
    assert isinstance(stats, dict)
    assert all(key in stats for key in ['mean', 'std', 'min', 'max'])
    assert stats['count'] == len(sample_ts)
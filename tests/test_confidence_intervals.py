import pytest
import pandas as pd
import numpy as np
from src.confidence.intervals import conformal_intervals_from_residuals, bootstrap_ci

def test_conformal_intervals():
    y_true = pd.Series(np.random.normal(100, 10, 100))
    y_pred = y_true + np.random.normal(0, 5, 100)
    future_pred = pd.Series(np.random.normal(100, 10, 30))
    
    ci = conformal_intervals_from_residuals(y_true, y_pred, future_pred)
    assert isinstance(ci, pd.DataFrame)
    assert all(col in ci.columns for col in ['lower', 'upper'])
    assert (ci['upper'] >= ci['lower']).all()
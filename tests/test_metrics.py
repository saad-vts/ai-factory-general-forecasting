import pytest
import numpy as np
from src.backtesting.evaluation import mape, smape, wmape, rmse
from src.metrics.selector import select_error_metrics

def test_mape():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([90, 210, 290])
    error = mape(y_true, y_pred)
    assert isinstance(error, float)
    assert error > 0

def test_metric_selection(sample_ts):
    metrics, primary = select_error_metrics(sample_ts)
    assert isinstance(metrics, list)
    assert primary in metrics
    assert len(metrics) >= 1
import numpy as np
import pandas as pd
from typing import List, Tuple
from src.backtesting.evaluation import mape, smape, wmape, rmse

def select_error_metrics(y: pd.Series) -> Tuple[List[str], str]:
    """
    Dynamically select appropriate error metrics based on data characteristics.
    
    Returns:
        tuple: (list of metric names to compute, primary metric for model selection)
    """
    # Analyze data characteristics
    zero_frac = (y == 0).mean()
    near_zero_frac = (y < y.mean() * 0.1).mean()
    cv = y.std() / y.mean() if y.mean() > 0 else float('inf')
    outlier_ratio = np.percentile(y, 95) / np.percentile(y, 50)
    
    metrics = []
    
    # If many zeros or near-zero values, MAPE is problematic
    if zero_frac > 0.1 or near_zero_frac > 0.2:
        metrics.append('smape')  # sMAPE handles zeros better
        primary = 'smape'
    # If high variability or outliers, consider RMSE
    elif cv > 1.5 or outlier_ratio > 5:
        metrics.append('rmse')
        primary = 'rmse'
    # If values are well-behaved, use MAPE
    else:
        metrics.append('mape')
        primary = 'mape'
    
    # Always include wMAPE as a supplementary metric
    if 'wmape' not in metrics:
        metrics.append('wmape')
    
    return metrics, primary

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_names: List[str]) -> dict:
    """Compute all requested error metrics"""
    results = {}
    
    for metric in metric_names:
        if metric == 'mape':
            results['mape'] = mape(y_true, y_pred)
        elif metric == 'smape':
            results['smape'] = smape(y_true, y_pred)
        elif metric == 'wmape':
            results['wmape'] = wmape(y_true, y_pred)
        elif metric == 'rmse':
            results['rmse'] = rmse(y_true, y_pred)
            
    return results
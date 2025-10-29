import numpy as np

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")

def smape(y_true, y_pred):
    a = np.abs(np.asarray(y_true)); b = np.abs(np.asarray(y_pred))
    denom = (a + b) / 2
    mask = denom != 0
    return float(np.mean(np.abs(a - b)[mask] / denom[mask]) * 100) if mask.any() else float("nan")

def wmape(y_true, y_pred, weights=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if weights is None:
        weights = np.abs(y_true)
    return float(np.sum(np.abs(y_true - y_pred) * weights) / max(1e-9, np.sum(weights))) * 100

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))
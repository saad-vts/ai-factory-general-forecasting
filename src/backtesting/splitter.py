from typing import Iterator, Tuple
import pandas as pd

def rolling_origin_splits(
    series: pd.Series,
    initial_train_days: int = 200,
    horizon: int = 30,
    stride: int = 30,
    min_train_size: int = 100
) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate rolling-origin split indices for time series cross validation.
    
    Args:
        series: Time series to split
        initial_train_days: Number of days for first training set
        horizon: Forecast horizon (validation size)
        stride: Number of days to move forward between splits
        min_train_size: Minimum required training days
        
    Yields:
        Tuple of (train_idx, val_idx) for each split
    """
    last = len(series)
    start = initial_train_days
    
    while start + horizon <= last:
        if start < min_train_size:
            break
            
        train_idx = series.index[:start]
        val_idx = series.index[start:start + horizon]
        
        yield train_idx, val_idx
        start += stride
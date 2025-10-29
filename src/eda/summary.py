import pandas as pd
import numpy as np
from typing import Dict, Any

def get_basic_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Calculate basic summary statistics for a time series.
    
    Args:
        series: pandas Series with datetime index
        
    Returns:
        Dictionary of statistics including:
        - count, mean, std, min, max
        - percentiles (25, 50, 75)
        - coefficient of variation
        - zero counts and percentage
    """
    stats = {}
    stats['count'] = len(series)
    stats['mean'] = float(series.mean())
    stats['std'] = float(series.std())
    stats['min'] = float(series.min())
    stats['max'] = float(series.max())
    stats['median'] = float(series.median())
    stats['p25'] = float(series.quantile(0.25))
    stats['p75'] = float(series.quantile(0.75))
    stats['cv'] = float(stats['std'] / max(abs(stats['mean']), 1e-8))
    stats['zeros_count'] = int((series == 0).sum())
    stats['zeros_pct'] = float(100 * stats['zeros_count'] / stats['count'])
    return stats

def get_temporal_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Calculate time-based statistics.
    
    Args:
        series: pandas Series with datetime index
        
    Returns:
        Dictionary with:
        - total span in days
        - average daily/weekly/monthly values
        - weekday/weekend averages
        - month-over-month growth rates
    """
    stats = {}
    stats['span_days'] = (series.index.max() - series.index.min()).days + 1
    
    # Daily/Weekly/Monthly averages
    stats['daily_avg'] = float(series.mean())
    stats['weekly_avg'] = float(series.resample('W').mean().mean())
    stats['monthly_avg'] = float(series.resample('M').mean().mean())
    
    # Day of week patterns
    dow_means = series.groupby(series.index.weekday).mean()
    stats['weekday_avg'] = float(dow_means[dow_means.index.isin([0,1,2,3,4])].mean())
    stats['weekend_avg'] = float(dow_means[dow_means.index.isin([5,6])].mean())
    
    # Month-over-month growth
    monthly = series.resample('M').mean()
    if len(monthly) > 1:
        mom_growth = monthly.pct_change().dropna()
        stats['mom_growth_avg'] = float(100 * mom_growth.mean())
        stats['mom_growth_std'] = float(100 * mom_growth.std())
    else:
        stats['mom_growth_avg'] = np.nan
        stats['mom_growth_std'] = np.nan
        
    return stats

def check_stationarity(series: pd.Series) -> Dict[str, Any]:
    """
    Basic stationarity checks using ADF test and rolling statistics.
    
    Args:
        series: pandas Series with datetime index
        
    Returns:
        Dictionary with stationarity test results and metrics
    """
    from statsmodels.tsa.stattools import adfuller
    
    results = {}
    # ADF test
    try:
        adf_result = adfuller(series.dropna())
        results['adf_pvalue'] = float(adf_result[1])
        results['adf_statistic'] = float(adf_result[0])
    except Exception:
        results['adf_pvalue'] = np.nan
        results['adf_statistic'] = np.nan
    
    # Rolling statistics
    roll = pd.DataFrame({
        'mean': series.rolling(window=30, min_periods=1).mean(),
        'std': series.rolling(window=30, min_periods=1).std()
    })
    
    results['rolling_mean_cv'] = float(roll['mean'].std() / roll['mean'].mean())
    results['rolling_std_cv'] = float(roll['std'].std() / roll['std'].mean())
    
    return results

def get_full_summary(df: pd.DataFrame, target_col: str = 'y') -> Dict[str, Any]:
    """
    Generate comprehensive EDA summary for a time series DataFrame.
    
    Args:
        df: pandas DataFrame with datetime index
        target_col: name of target column (default: 'y')
        
    Returns:
        Dictionary with all summary statistics
    """
    series = df[target_col]
    
    summary = {
        'basic_stats': get_basic_stats(series),
        'temporal_stats': get_temporal_stats(series),
        'stationarity': check_stationarity(series)
    }
    
    # Add data quality metrics
    summary['quality'] = {
        'missing_count': int(series.isna().sum()),
        'missing_pct': float(100 * series.isna().mean()),
        'unique_count': int(series.nunique()),
        'unique_pct': float(100 * series.nunique() / len(series))
    }
    
    return summary
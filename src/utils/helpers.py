import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any

def tsplot(
    series: Union[pd.Series, pd.DataFrame],
    title: Optional[str] = None,
    figsize: tuple = (12, 4),
    ma_window: int = 7,
    ci: Optional[pd.DataFrame] = None
) -> None:
    """
    Plot time series data with optional moving average and confidence intervals.
    
    Args:
        series: Time series data to plot
        title: Optional plot title
        figsize: Figure size tuple
        ma_window: Moving average window size
        ci: Optional DataFrame with 'lower' and 'upper' columns for confidence intervals
    """
    plt.figure(figsize=figsize)
    
    if isinstance(series, pd.DataFrame):
        for col in series.columns:
            plt.plot(series.index, series[col], alpha=0.7, label=col)
    else:
        plt.plot(series.index, series.values, alpha=0.7, label='Actual')
        
        # Add moving average
        ma = series.rolling(window=ma_window, min_periods=1).mean()
        plt.plot(ma.index, ma.values, 
                linewidth=2, label=f'{ma_window}d MA')
    
    # Add confidence intervals if provided
    if ci is not None and 'lower' in ci.columns and 'upper' in ci.columns:
        plt.fill_between(ci.index, ci['lower'], ci['upper'],
                        alpha=0.2, color='gray', label='95% CI')
    
    if title:
        plt.title(title)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_forecast_results(df: pd.DataFrame, forecast_df: pd.DataFrame, 
                         model_name: str, horizon: int, outdir: Path):
    """
    Plot historical data with forecast and confidence intervals.
    
    Args:
        df: Historical data
        forecast_df: Forecast DataFrame with 'forecast', 'lower', 'upper'
        model_name: Name of the model used
        horizon: Forecast horizon in periods
        outdir: Output directory for saving plots
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Determine how much history to show based on frequency
    freq = pd.infer_freq(df.index)
    if freq and freq.startswith('M'):
        lookback = min(36, len(df))  # Last 3 years for monthly
    elif freq and freq.startswith('W'):
        lookback = min(52, len(df))  # Last year for weekly
    else:
        lookback = min(180, len(df))  # Last 6 months for daily
    
    # Plot historical data
    history = df['y'].iloc[-lookback:]
    ax.plot(history.index, history.values, 'b-', label='Historical', linewidth=2)
    
    # Validate forecast_df has datetime index
    if not isinstance(forecast_df.index, pd.DatetimeIndex):
        print(f"Warning: forecast_df index is not DatetimeIndex, got {type(forecast_df.index)}")
        return
    
    # Plot forecast
    ax.plot(forecast_df.index, forecast_df['forecast'], 'r-', 
            label='Forecast', linewidth=2, marker='o', markersize=4)
    
    # Plot confidence intervals if available
    if 'lower' in forecast_df.columns and 'upper' in forecast_df.columns:
        ax.fill_between(forecast_df.index, 
                        forecast_df['lower'], 
                        forecast_df['upper'],
                        alpha=0.3, color='red', label='95% CI')
    
    # Format title based on frequency
    freq_label = "Period" if not freq else ("Month" if freq.startswith('M') else ("Week" if freq.startswith('W') else "Day"))
    
    ax.set_title(f'{model_name.upper()} - {horizon}-{freq_label} Forecast', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plot_path = outdir / f'{model_name}_forecast_{horizon}p.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

def plot_model_comparison(results: Dict[str, Any], outdir: Path, primary_metric: str):
    """
    Plot comparison of different models' performance.
    
    Args:
        results: Dictionary with model results
        outdir: Output directory
        primary_metric: Primary metric being used
    """
    models = []
    scores = []
    
    for key, value in results.items():
        if key.endswith('_metrics') and isinstance(value, dict):
            model_name = key.replace('_metrics', '')
            if primary_metric in value:
                models.append(model_name.replace('_', ' ').title())
                scores.append(value[primary_metric])
    
    if not models:
        print("No model metrics available for comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(models, scores, color=colors[:len(models)])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(f'Model Comparison - {primary_metric.upper()}', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel(primary_metric.upper(), fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = outdir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

def plot_diagnostics(df: pd.DataFrame, outdir: Path):
    """
    Plot diagnostic visualizations of the time series.
    
    Args:
        df: DataFrame with time series data
        outdir: Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    axes[0, 0].plot(df.index, df['y'], 'b-', linewidth=1)
    axes[0, 0].set_title('Time Series', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distribution
    axes[0, 1].hist(df['y'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Autocorrelation (simple version)
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(df['y'].dropna(), ax=axes[1, 0])
    axes[1, 0].set_title('Autocorrelation', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Monthly boxplot
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly.index.month
    df_monthly.boxplot(column='y', by='month', ax=axes[1, 1])
    axes[1, 1].set_title('Monthly Patterns', fontweight='bold')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Value')
    plt.suptitle('')  # Remove default title
    
    plt.tight_layout()
    plot_path = outdir / 'diagnostics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

def plot_residuals(y_true: pd.Series, y_pred: pd.Series, model_name: str, outdir: Path):
    """
    Plot residual diagnostics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of model
        outdir: Output directory
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals over time
    axes[0, 0].plot(y_true.index, residuals, 'b-', alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[0, 1].hist(residuals.dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Residual Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter: Predicted vs Actual
    axes[1, 1].scatter(y_pred, y_true, alpha=0.5)
    axes[1, 1].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 'r--')
    axes[1, 1].set_title('Predicted vs Actual', fontweight='bold')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = outdir / f'{model_name}_residuals.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

# These should be at module level, not inside plot_residuals
def plot_learning_curves(train_scores: list, valid_scores: list, 
                        model_name: str, outdir: Path):
    """
    Plot learning curves to visualize overfitting.
    
    Args:
        train_scores: Training scores over iterations
        valid_scores: Validation scores over iterations
        model_name: Name of the model
        outdir: Output directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(1, len(train_scores) + 1)
    ax.plot(iterations, train_scores, 'b-', label='Training', linewidth=2)
    ax.plot(iterations, valid_scores, 'r-', label='Validation', linewidth=2)
    
    # Mark best iteration
    best_iter = np.argmin(valid_scores) + 1
    ax.axvline(x=best_iter, color='g', linestyle='--', 
              label=f'Best Iteration ({best_iter})')
    
    ax.set_title(f'{model_name.upper()} - Learning Curves', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = outdir / f'{model_name}_learning_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

def detect_overfitting(train_metric: float, valid_metric: float, 
                      threshold: float = 0.2) -> tuple:
    """
    Detect overfitting by comparing train and validation metrics.
    
    Args:
        train_metric: Training metric (lower is better)
        valid_metric: Validation metric (lower is better)
        threshold: Acceptable degradation threshold
    
    Returns:
        Tuple of (is_overfitting, message)
    """
    if train_metric == 0:
        return False, "Cannot assess - zero training metric"
    
    degradation = (valid_metric - train_metric) / train_metric
    
    if degradation > threshold:
        return True, f"Overfitting detected: {degradation:.1%} degradation"
    else:
        return False, f"Good fit: {degradation:.1%} degradation"
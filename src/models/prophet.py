import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import warnings

def fit_prophet(
    df: pd.DataFrame,
    freq: Optional[str] = None,
    budget_sec: int = 90,
    enable_cv: bool = True,
    holidays: Optional[pd.DataFrame] = None,
    country: str = "US"
) -> Optional[object]:
    """
    Fit a Prophet model with timeout budget and optional cross-validation.
    Robust for any frequency: hourly, daily, weekly, monthly, quarterly, yearly.
    
    Args:
        df: DataFrame with 'ds' (datetime) and 'y' (numeric) columns
        freq: Frequency string (optional, will infer if not provided)
        budget_sec: Time budget for training in seconds
        enable_cv: Whether to enable cross-validation
        holidays: Optional holidays dataframe
        country: Country code for holidays
    
    Returns:
        Fitted Prophet model or None if failed
    """
    try:
        from prophet import Prophet
        from prophet.diagnostics import cross_validation, performance_metrics
        import time
        
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError(f"DataFrame must have 'ds' and 'y' columns. Got: {df.columns.tolist()}")
        
        if len(df) < 2:
            raise ValueError(f"Need at least 2 observations, got {len(df)}")
        
        # Ensure proper types
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])
        
        # Remove any NaN values
        df = df.dropna(subset=['ds', 'y'])
        
        if len(df) < 2:
            raise ValueError("Not enough valid data after removing NaN values")
        
        # Infer frequency if not provided
        if freq is None:
            if len(df) >= 2:
                freq_diff = df['ds'].diff().median()
                if freq_diff <= pd.Timedelta(hours=1):
                    freq = 'h'
                elif freq_diff <= pd.Timedelta(days=1):
                    freq = 'D'
                elif freq_diff <= pd.Timedelta(days=7):
                    freq = 'W'
                elif freq_diff <= pd.Timedelta(days=31):
                    freq = 'MS'
                elif freq_diff <= pd.Timedelta(days=92):
                    freq = 'QS'
                else:
                    freq = 'YS'
            else:
                freq = 'D'
        
        print(f"    Prophet training data: {len(df)} periods from {df['ds'].min()} to {df['ds'].max()}")
        print(f"    Frequency: {freq}")
        
        # Set CV parameters based on frequency
        # Use DateOffset instead of Timedelta for months/quarters/years
        if freq in ('h', 'H'):
            # Hourly
            horizon = '7 days'
            initial = '14 days'
            period = '3 days'
            cv_cutoff = pd.Timedelta(days=21)
            
        elif freq in ('D', 'B'):
            # Daily
            horizon = '30 days'
            initial = '90 days'
            period = '7 days'
            cv_cutoff = pd.Timedelta(days=120)
            
        elif freq in ('W', 'W-SUN'):
            # Weekly
            horizon = '90 days'
            initial = '365 days'
            period = '30 days'
            cv_cutoff = pd.Timedelta(days=455)
            
        elif freq in ('MS', 'M', 'ME'):
            # Monthly - use DateOffset for months
            horizon = '365 days'  # ~12 months
            initial = '730 days'  # ~24 months
            period = '180 days'   # ~6 months
            cv_cutoff = pd.DateOffset(months=30)
            
        elif freq in ('QS', 'Q', 'QE'):
            # Quarterly
            horizon = '365 days'  # 4 quarters
            initial = '1095 days'  # 12 quarters (3 years)
            period = '365 days'   # 4 quarters
            cv_cutoff = pd.DateOffset(months=48)
            
        elif freq in ('YS', 'Y', 'YE'):
            # Yearly
            horizon = '1095 days'  # 3 years
            initial = '3650 days'  # 10 years
            period = '1095 days'   # 3 years
            cv_cutoff = pd.DateOffset(years=13)
            
        else:
            # Default to daily
            horizon = '30 days'
            initial = '90 days'
            period = '7 days'
            cv_cutoff = pd.Timedelta(days=120)
        
        # Check if we have enough data for CV
        data_span = df['ds'].max() - df['ds'].min()
        
        # Convert cv_cutoff to comparable format
        if isinstance(cv_cutoff, pd.DateOffset):
            # For DateOffset, check against data span in days
            can_do_cv = len(df) > 50  # Simpler check for monthly/quarterly/yearly
        else:
            can_do_cv = data_span >= cv_cutoff and len(df) > 50
        
        print(f"    Data span: {data_span}, CV enabled: {can_do_cv}")
        
        start = time.time()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Try different regularization levels if CV enabled
            if enable_cv and can_do_cv:
                best_model = None
                best_mape = float('inf')
                
                # Test different changepoint_prior_scale values
                for cps in [0.001, 0.01, 0.05, 0.5]:
                    if time.time() - start > budget_sec * 0.8:
                        print(f"      Stopping CV due to time budget")
                        break
                    
                    model = Prophet(
                        daily_seasonality=True if freq in ['h', 'H', 'D', 'B'] else False,
                        weekly_seasonality='auto' if freq in ['h', 'H', 'D', 'B'] else False,
                        yearly_seasonality='auto',
                        changepoint_prior_scale=cps,
                        seasonality_prior_scale=10.0,
                        seasonality_mode='multiplicative',
                        interval_width=0.95,
                        holidays=holidays
                    )
                    
                    model.fit(df)
                    
                    # Cross-validate
                    try:
                        df_cv = cross_validation(
                            model,
                            initial=initial,
                            period=period,
                            horizon=horizon,
                            parallel="processes"
                        )
                        df_p = performance_metrics(df_cv)
                        mape = df_p['mape'].mean()
                        
                        print(f"      CPS={cps:.3f} → CV MAPE: {mape:.2f}")
                        
                        if mape < best_mape:
                            best_mape = mape
                            best_model = model
                            
                    except Exception as e:
                        print(f"      CPS={cps:.3f} → CV failed: {str(e)}")
                        continue
                
                if best_model is None:
                    print(f"    All CV attempts failed, using default model")
                    best_model = Prophet(
                        daily_seasonality=True if freq in ['h', 'H', 'D', 'B'] else False,
                        weekly_seasonality='auto' if freq in ['h', 'H', 'D', 'B'] else False,
                        yearly_seasonality='auto',
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=10.0,
                        seasonality_mode='multiplicative',
                        interval_width=0.95,
                        holidays=holidays
                    )
                    best_model.fit(df)
                else:
                    print(f"    ✓ Best model: CPS={best_model.changepoint_prior_scale:.3f}, CV MAPE={best_mape:.2f}")
                
                model = best_model
            else:
                if not can_do_cv:
                    print(f"    Skipping CV: insufficient data (need {len(df) > 50} rows and adequate span)")
                
                # Default model without CV
                model = Prophet(
                    daily_seasonality=True if freq in ['h', 'H', 'D', 'B'] else False,
                    weekly_seasonality='auto' if freq in ['h', 'H', 'D', 'B'] else False,
                    yearly_seasonality='auto',
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    seasonality_mode='multiplicative',
                    interval_width=0.95,
                    holidays=holidays
                )
                
                model.fit(df)
            
            # Store frequency and training info as attributes
            model.freq = freq
            model.last_train_date = df['ds'].max()
            model.n_train = len(df)
        
        elapsed = time.time() - start
        print(f"✓ Prophet fitted in {elapsed:.1f}s")
        
        if elapsed > budget_sec:
            print(f"    ⚠️ Warning: Training exceeded budget ({elapsed:.1f}s > {budget_sec}s)")
        
        return model
        
    except Exception as e:
        print(f"    ✗ Prophet fitting failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def forecast_prophet(
    model,
    horizon: int,
    freq: Optional[str] = None,
    last_date: Optional[pd.Timestamp] = None
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Generate forecast using Prophet model.
    Handles both attribute-based and parameter-based configurations.
    Robust against date overlap issues and various frequency formats.
    
    Args:
        model: Fitted Prophet model
        horizon: Number of periods to forecast
        freq: Frequency string (optional, will use model attribute if not provided)
        last_date: Last training date (optional, for logging)
    
    Returns:
        forecast_values: Array of forecasted values
        confidence: Dict with 'lower' and 'upper' confidence bounds
    """
    try:
        # Get frequency from parameter or model attribute
        if freq is None:
            freq = getattr(model, 'freq', 'D')
        
        # Get training info from model attributes
        last_train_date = last_date or getattr(model, 'last_train_date', None)
        n_train = getattr(model, 'n_train', 0)
        
        # Detect if sub-daily
        is_subdaily = freq in ["h", "H", "15min", "30min", "min", "5min"]
        
        print(f"    Generating {horizon} periods forecast")
        if last_train_date:
            print(f"    Last training date: {last_train_date}")
        print(f"    Frequency: {freq} (sub-daily: {is_subdaily})")
        
        # Create future dataframe with correct frequency
        if is_subdaily:
            # Map frequency strings to pandas-compatible formats
            freq_map = {
                'h': 'h',
                'H': 'h',
                '15min': '15min',
                '30min': '30min',
                'min': 'min',
                '5min': '5min'
            }
            pandas_freq = freq_map.get(freq, freq)
            
            future = model.make_future_dataframe(
                periods=horizon,
                freq=pandas_freq,
                include_history=False  # CRITICAL: Only future periods
            )
        else:
            future = model.make_future_dataframe(
                periods=horizon,
                freq=freq,
                include_history=False
            )
        
        print(f"    Future dataframe: {len(future)} rows")
        print(f"    Future range: {future['ds'].min()} to {future['ds'].max()}")
        
        # Validate future dates are actually in the future
        if last_train_date:
            if future['ds'].min() <= last_train_date:
                print(f"    ⚠️ WARNING: Future dates overlap with training!")
                print(f"    Training ends: {last_train_date}")
                print(f"    Future starts: {future['ds'].min()}")
                
                # Filter to only actual future dates
                future = future[future['ds'] > last_train_date].copy()
                print(f"    Filtered to {len(future)} truly future periods")
                
                if len(future) == 0:
                    print(f"    ✗ Error: No future periods after filtering")
                    return None, None
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract forecasts (should match future dataframe)
        forecast_values = forecast['yhat'].values
        lower = forecast['yhat_lower'].values
        upper = forecast['yhat_upper'].values
        forecast_dates = forecast['ds'].values
        
        print(f"    ✓ Generated {len(forecast_values)} forecasts")
        print(f"    Forecast date range: {forecast_dates[0]} to {forecast_dates[-1]}")
        
        # Ensure we have exactly `horizon` forecasts
        if len(forecast_values) != horizon:
            print(f"    ⚠️ Warning: Expected {horizon} forecasts, got {len(forecast_values)}")
            
            if len(forecast_values) > horizon:
                # Trim to horizon
                forecast_values = forecast_values[:horizon]
                lower = lower[:horizon]
                upper = upper[:horizon]
                print(f"    Trimmed to {horizon} periods")
                
            elif len(forecast_values) < horizon:
                # Accept if within 10% of target
                shortfall = horizon - len(forecast_values)
                if shortfall < horizon * 0.1:
                    print(f"    Accepting {len(forecast_values)} periods (within 10% of target)")
                else:
                    # Pad with last value
                    forecast_values = np.pad(forecast_values, (0, shortfall), mode='edge')
                    lower = np.pad(lower, (0, shortfall), mode='edge')
                    upper = np.pad(upper, (0, shortfall), mode='edge')
                    print(f"    Padded to {horizon} periods")
        
        # For data with zeros (like traffic), ensure non-negative forecasts
        forecast_values = np.clip(forecast_values, 0, None)
        lower = np.clip(lower, 0, None)
        upper = np.clip(upper, 0, None)
        
        confidence = {
            'lower': lower,
            'upper': upper
        }
        
        print(f"    Value range: {forecast_values.min():.2f} to {forecast_values.max():.2f}")
        
        # Final validation
        if len(forecast_values) == 0 or len(lower) == 0 or len(upper) == 0:
            print(f"    ✗ Error: Empty arrays after processing")
            return None, None
        
        return forecast_values, confidence
        
    except Exception as e:
        print(f"    ✗ Prophet forecasting failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
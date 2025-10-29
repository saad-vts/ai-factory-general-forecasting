
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import warnings

def fit_prophet(df: pd.DataFrame, budget_sec: int = 90, 
               enable_cv: bool = True) -> Optional[object]:
    """
    Fit a Prophet model with timeout budget and optional cross-validation.
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
        
        # Infer frequency
        if len(df) >= 2:
            freq_diff = df['ds'].diff().median()
            if freq_diff <= pd.Timedelta(hours=1):
                freq = 'H'  # Hourly or sub-hourly
                horizon_days = '7 days'
                initial_days = '14 days'
                period_days = '3 days'
            elif freq_diff >= pd.Timedelta(days=28):
                freq = 'MS'  # Month Start
                horizon_days = '365 days'
                initial_days = '730 days'
                period_days = '180 days'
            elif freq_diff >= pd.Timedelta(days=7):
                freq = 'W'
                horizon_days = '90 days'
                initial_days = '365 days'
                period_days = '30 days'
            else:
                freq = 'D'
                horizon_days = '30 days'
                initial_days = '90 days'
                period_days = '7 days'
        else:
            freq = 'D'
            horizon_days = '30 days'
            initial_days = '90 days'
            period_days = '7 days'
        
        print(f"    Inferred frequency: {freq}")
        print(f"    Training data range: {df['ds'].min()} to {df['ds'].max()}")
        
        # Check if we have enough data for CV
        data_span = df['ds'].max() - df['ds'].min()
        initial_td = pd.Timedelta(initial_days)
        horizon_td = pd.Timedelta(horizon_days)
        
        can_do_cv = data_span >= (initial_td + horizon_td) and len(df) > 50
        
        start = time.time()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Try different regularization levels if CV enabled
            if enable_cv and can_do_cv:
                best_model = None
                best_mape = float('inf')
                
                # Test different changepoint_prior_scale values
                for cps in [0.001, 0.01, 0.05, 0.5]:
                    model = Prophet(
                        daily_seasonality=True if freq in ['H', 'D'] else False,
                        weekly_seasonality='auto' if freq in ['H', 'D'] else False,
                        yearly_seasonality='auto',
                        changepoint_prior_scale=cps,
                        seasonality_prior_scale=10.0,
                        seasonality_mode='multiplicative',
                        interval_width=0.95
                    )
                    
                    model.fit(df)
                    
                    # Cross-validate
                    try:
                        df_cv = cross_validation(
                            model, 
                            initial=initial_days,
                            period=period_days, 
                            horizon=horizon_days,
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
                    
                    # Check time budget
                    if time.time() - start > budget_sec * 0.8:
                        print(f"      Stopping CV due to time budget")
                        break
                
                if best_model is None:
                    print(f"    All CV attempts failed, using default model")
                    best_model = Prophet(
                        daily_seasonality=True if freq in ['H', 'D'] else False,
                        weekly_seasonality='auto' if freq in ['H', 'D'] else False,
                        yearly_seasonality='auto',
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=10.0,
                        interval_width=0.95
                    )
                    best_model.fit(df)
                else:
                    print(f"    ✓ Best model: CPS={best_model.changepoint_prior_scale:.3f}, CV MAPE={best_mape:.2f}")
                
                model = best_model
            else:
                if not can_do_cv:
                    print(f"    Skipping CV: insufficient data span ({data_span} < {initial_td + horizon_td})")
                
                # Default model without CV
                model = Prophet(
                    daily_seasonality=True if freq in ['H', 'D'] else False,
                    weekly_seasonality='auto' if freq in ['H', 'D'] else False,
                    yearly_seasonality='auto',
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    interval_width=0.95
                )
                
                model.fit(df)
            
            # Store frequency and training info
            model.freq = freq
            model.last_train_date = df['ds'].max()
            model.n_train = len(df)
        
        elapsed = time.time() - start
        print(f"    Prophet fitted in {elapsed:.1f}s")
        
        if elapsed > budget_sec:
            print(f"    ⚠️ Warning: Training exceeded budget ({elapsed:.1f}s > {budget_sec}s)")
        
        return model
        
    except Exception as e:
        print(f"    Prophet fitting failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def forecast_prophet(model, horizon: int) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Generate forecast using Prophet model.
    """
    try:
        # Get frequency and training info from model
        freq = getattr(model, 'freq', 'D')
        last_train_date = getattr(model, 'last_train_date', None)
        n_train = getattr(model, 'n_train', 0)
        
        print(f"    Generating {horizon} periods forecast starting after {last_train_date}")
        
        # Create future dataframe with correct frequency
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        
        print(f"    Future dataframe: {len(future)} rows, range: {future['ds'].min()} to {future['ds'].max()}")
        print(f"    n_train: {n_train}, total future rows: {len(future)}")
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract ONLY the future forecasts (after training data)
        if n_train > 0 and n_train < len(forecast):
            forecast_values = forecast['yhat'].iloc[n_train:].values
            lower = forecast['yhat_lower'].iloc[n_train:].values
            upper = forecast['yhat_upper'].iloc[n_train:].values
            forecast_dates = forecast['ds'].iloc[n_train:].values
            
            print(f"    Extracted forecasts from index {n_train} to {len(forecast)} ({len(forecast_values)} periods)")
        else:
            # Fallback: take last `horizon` periods
            forecast_values = forecast['yhat'].iloc[-horizon:].values
            lower = forecast['yhat_lower'].iloc[-horizon:].values
            upper = forecast['yhat_upper'].iloc[-horizon:].values
            forecast_dates = forecast['ds'].iloc[-horizon:].values
            print(f"    Fallback: using last {horizon} periods")
        
        if len(forecast_values) == 0:
            print(f"    ⚠️ Error: Empty forecast array. n_train={n_train}, total={len(forecast)}, horizon={horizon}")
            return None, None
        
        print(f"    Forecast range: {forecast_dates[0]} to {forecast_dates[-1]} ({len(forecast_values)} periods)")
        
        # Verify we have the right number of forecasts
        if len(forecast_values) != horizon:
            print(f"    ⚠️ Warning: Expected {horizon} forecasts, got {len(forecast_values)}")
            # Trim or pad to match horizon
            if len(forecast_values) > horizon:
                forecast_values = forecast_values[:horizon]
                lower = lower[:horizon]
                upper = upper[:horizon]
        
        confidence = {
            'lower': lower[:horizon],
            'upper': upper[:horizon]
        }
        
        return forecast_values, confidence
        
    except Exception as e:
        print(f"    Prophet forecasting failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
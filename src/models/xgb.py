import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict

def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, 
              X_valid: pd.DataFrame, y_valid: pd.Series,
              budget_sec: int = 90) -> Tuple[Optional[xgb.Booster], Optional[Dict]]:
    """
    Train XGBoost model with regularization and early stopping to prevent overfitting.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_valid: Validation features
        y_valid: Validation target
        budget_sec: Time budget in seconds
    
    Returns:
        Tuple of (trained model, metrics dict) or (None, None) if failed
    """
    try:
        import time
        
        # Validate inputs
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}")
        
        if len(X_valid) != len(y_valid):
            raise ValueError(f"X_valid and y_valid length mismatch: {len(X_valid)} vs {len(y_valid)}")
        
        start = time.time()
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        
        # Parameters with regularization to control overfitting
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,  # Shallow trees = less overfitting
            'learning_rate': 0.05,  # Lower learning rate = more robust
            'subsample': 0.8,  # Row sampling
            'colsample_bytree': 0.8,  # Feature sampling
            'colsample_bylevel': 0.8,  # Feature sampling per level
            'min_child_weight': 3,  # Minimum samples per leaf
            'gamma': 0.1,  # Minimum loss reduction for split
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'seed': 42,
            'verbosity': 0
        }
        
        print(f"    XGBoost params: max_depth={params['max_depth']}, "
              f"lr={params['learning_rate']}, "
              f"l1={params['reg_alpha']}, l2={params['reg_lambda']}")
        
        # Train with early stopping
        evals = [(dtrain, 'train'), (dvalid, 'valid')]
        evals_result = {}
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,  # Max iterations
            evals=evals,
            early_stopping_rounds=20,  # Stop if no improvement for 20 rounds
            evals_result=evals_result,
            verbose_eval=False
        )
        
        elapsed = time.time() - start
        best_iteration = model.best_iteration
        print(f"    XGBoost trained in {elapsed:.1f}s (stopped at iteration {best_iteration})")
        
        # Check for overfitting
        train_rmse = evals_result['train']['rmse'][-1]
        valid_rmse = evals_result['valid']['rmse'][-1]
        overfit_ratio = valid_rmse / train_rmse if train_rmse > 0 else 1.0
        
        if overfit_ratio > 1.5:
            print(f"    ⚠️ Warning: Potential overfitting detected (valid/train RMSE ratio: {overfit_ratio:.2f})")
        else:
            print(f"    ✓ Good fit (valid/train RMSE ratio: {overfit_ratio:.2f})")
        
        # Get metrics
        pred_valid = model.predict(dvalid)
        from src.backtesting.evaluation import mape, smape
        
        metrics = {
            'mape': mape(y_valid.values, pred_valid),
            'smape': smape(y_valid.values, pred_valid),
            'train_rmse': float(train_rmse),
            'valid_rmse': float(valid_rmse),
            'overfit_ratio': float(overfit_ratio),
            'n_iterations': int(best_iteration)
        }
        
        return model, metrics
        
    except Exception as e:
        print(f"XGBoost training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def xgb_recursive_forecast(hist: pd.Series, model: xgb.Booster, 
                          feat_func, horizon: int) -> Optional[pd.Series]:
    """
    Generate recursive forecasts using XGBoost.
    
    Args:
        hist: Historical series
        model: Trained XGBoost model
        feat_func: Function to generate features for a date
        horizon: Forecast horizon
    
    Returns:
        Series of forecasts or None if failed
    """
    try:
        forecast_dates = pd.date_range(
            hist.index[-1] + pd.Timedelta(days=1), 
            periods=horizon, 
            freq='D'
        )
        forecasts = []
        current_series = hist.copy()
        
        for date in forecast_dates:
            # Generate features for current date
            feats = feat_func(horizon, date)
            
            # Make prediction
            dtest = xgb.DMatrix(feats)
            pred = model.predict(dtest)[0]
            forecasts.append(pred)
            
            # Append to series for next iteration
            new_point = pd.Series([pred], index=[date])
            current_series = pd.concat([current_series, new_point])
        
        return pd.Series(forecasts, index=forecast_dates)
        
    except Exception as e:
        print(f"XGBoost recursive forecast failed: {str(e)}")
        return None
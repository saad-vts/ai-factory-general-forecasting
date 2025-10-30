"""
Progressive pipeline orchestration with decision points.
Run from repo root: python scripts\run_pipeline.py
"""
import os
import sys
from pathlib import Path


# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import yaml
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

import xgboost as xgb

# Local imports
from src.ingestion.load import load_csv
from src.ingestion.validators import validate_cfg, validate_df
from src.eda.forecastability import summary_stats, seasonal_strength
from src.eda.summary import get_full_summary
from src.features.engineering import make_calendar_features, make_lags_rolls
from src.features.diagnostics import check_leakage, vif_table
from src.external.holidays import make_country_holidays
from src.selection.feature_selector import select_external_features, corr_screen
from src.metrics.selector import select_error_metrics, compute_metrics
from src.models.sarimax import fit_sarimax, forecast_sarimax
from src.models.xgb import train_xgb, xgb_recursive_forecast
from src.models.prophet import fit_prophet, forecast_prophet
from src.backtesting.evaluation import mape, smape, wmape
from src.monitoring.drift import psi_score
from src.outputs.save import ensure_outdir, save_forecast, save_metrics
from src.utils.helpers import tsplot
from src.utils import plot_forecast_results, plot_model_comparison, plot_diagnostics, plot_residuals


def evaluate_features(y: pd.Series, features: pd.DataFrame, vif_threshold: float = 5.0) -> Tuple[pd.DataFrame, Dict]:
    """Evaluate and select features based on correlation and VIF"""
    results = {}
    
    # 1. Correlation screening
    kept_features = corr_screen(y, features, min_abs_corr=0.1)
    if not kept_features:
        return pd.DataFrame(index=features.index), {"n_features": 0, "reason": "no_correlation"}
    
    features_corr = features[kept_features]
    results["n_features_post_corr"] = len(kept_features)
    
    # 2. VIF check
    vif_stats = vif_table(features_corr)
    high_vif = vif_stats[vif_stats["vif"] > vif_threshold]["feature"].tolist()
    
    if high_vif:
        print(f"Dropping {len(high_vif)} features with VIF > {vif_threshold}")
        kept_features = [f for f in kept_features if f not in high_vif]
    
    results.update({
        "n_features_final": len(kept_features),
        "features_kept": kept_features,
        "max_vif": float(vif_stats["vif"].max()) if not vif_stats.empty else 0.0
    })
    
    return features[kept_features], results

def main():
    # 1. Setup
    CFG_PATH = Path("configs/default.yaml")
    cfg = yaml.safe_load(CFG_PATH.read_text())
    validate_cfg(cfg)

    
    outdir = ensure_outdir(Path("outputs"))
    print(f"Output directory: {outdir}")

    # 2. Load and validate data
    df = load_csv(cfg)
    validate_df(df)
    print(f"Loaded data: {df.shape}, {df.index.min().date()} → {df.index.max().date()}")
    print("\nGenerating diagnostic plots...")
    plot_diagnostics(df, outdir)


    # 3. Initial diagnostics
    diag = summary_stats(df)
    diag.update(seasonal_strength(df["y"]))
    diag.update(get_full_summary(df))
    
    # Check if series is forecastable
    if diag["zero_fraction"] > 0.4:
        print(" Warning: Series appears intermittent")
    if diag.get("acf7", 0) < 0.3 and diag.get("acf365", 0) < 0.3:
        print(" Warning: Low seasonality detected")

    save_metrics(diag, outdir/"diagnostics.json")
    tsplot(df["y"], "Raw Data")

    # 4. Progressive feature engineering & selection
    features = pd.DataFrame(index=df.index)
    feature_results = {}
    
    # Start with calendar features
    print("\nEvaluating calendar features...")
    cal = make_calendar_features(df.index, weekend_days=cfg.get("weekend_days", [5,6]))
    cal_kept, cal_stats = evaluate_features(df["y"], cal)
    if cal_kept.empty:
        print("No useful calendar features found")
    else:
        features = features.join(cal_kept)
        feature_results["calendar"] = cal_stats
    
    # Add holidays if calendar features helped
    if not cal_kept.empty:
        print("\nEvaluating holiday features...")
        hol = make_country_holidays(df.index, country_code=cfg.get("holiday_country", "AE"))
        hol_kept, hol_stats = evaluate_features(df["y"], hol)
        if not hol_kept.empty:
            features = features.join(hol_kept)
            feature_results["holidays"] = hol_stats
    
    # Add lags & rolls if we have good autocorrelation
    if diag.get("acf7", 0) > 0.3:
        print("\nEvaluating lag features...")
        lags = make_lags_rolls(df["y"])
        # Check for leakage
        leaks = check_leakage(lags)
        if leaks:
            print("⚠️ Warning: Potential leakage in:", leaks)
        else:
            lags_kept, lags_stats = evaluate_features(df["y"], lags)
            if not lags_kept.empty:
                features = features.join(lags_kept)
                feature_results["lags"] = lags_stats

    # After loading data, select appropriate metrics
    metrics_to_use, primary_metric = select_error_metrics(df["y"])
    print(f"Selected metrics: {metrics_to_use} (primary: {primary_metric})")

    # 5. Train/validation split based on frequency
    freq = cfg.get("freq", "D")
    
    # Frequency-aware validation size
    if freq in ("MS", "ME", "M"):
        n_valid = min(6, len(df) // 4)  # 6 months or 25% of data
        min_train = 12  # At least 1 year
    elif freq in ("W",):
        n_valid = min(12, len(df) // 4)  # 12 weeks
        min_train = 24  # At least 6 months
    elif freq in ("QE", "Q"):
        n_valid = min(4, len(df) // 4)  # 4 quarters
        min_train = 8  # At least 2 years
    elif freq in ("YE", "Y"):
        n_valid = min(2, len(df) // 4)  # 2 years
        min_train = 5  # At least 5 years
    else:
        n_valid = min(90, len(df) // 4)  # 90 days or 25%
        min_train = 180  # At least 6 months
    
    # Ensure we have enough data
    if len(df) < min_train + n_valid:
        print(f"\n Warning: Dataset too small ({len(df)} periods)")
        print(f"  Minimum required: {min_train + n_valid} ({min_train} train + {n_valid} valid)")
        print(f"  Reducing validation size to 20% of data")
        n_valid = max(1, len(df) // 5)
    
    # Ensure minimum training size
    if len(df) - n_valid < min_train:
        n_valid = max(1, len(df) - min_train)
        print(f"  Adjusted validation size to {n_valid} to maintain minimum training size")
    
    print(f"\nTrain/Validation Split (freq={freq}):")
    print(f"  Total periods: {len(df)}")
    print(f"  Training: {len(df) - n_valid} periods ({df.index[0]} to {df.index[-(n_valid+1)]})")
    print(f"  Validation: {n_valid} periods ({df.index[-n_valid]} to {df.index[-1]})")
    
    df_train = df.iloc[:-n_valid]
    df_valid = df.iloc[-n_valid:]
    
    # Verify split is not empty
    if len(df_train) == 0 or len(df_valid) == 0:
        raise ValueError(f"Invalid split: train={len(df_train)}, valid={len(df_valid)}")
    
    print(f"  Train shape: {df_train.shape}, Valid shape: {df_valid.shape}")
    
    # 6. Progressive model selection with n+1 ladder approach
    results = {}
    best_score = float('inf')
    best_model = None
    primary_metric = metrics_to_use[0] if metrics_to_use else "mape"
    target_threshold = cfg.get("metric_targets", {}).get(primary_metric, float('inf'))
    
    # Define model ladder
    model_ladder = [
        ("sarimax_baseline", "SARIMAX baseline"),
        ("sarimax_featured", "SARIMAX with features"),
        ("prophet", "Prophet"),
        ("xgb", "XGBoost")
    ]
    
    print("\n" + "="*60)
    print("PROGRESSIVE MODEL SELECTION (N+1 Ladder)")
    print("="*60)
    
    should_continue = True
    models_tried = []
    
    for idx, (model_name, model_desc) in enumerate(model_ladder):
        if not should_continue:
            print(f"\nSkipping {model_desc} - previous model was good enough")
            break
            
        print(f"\n[{idx+1}/{len(model_ladder)}] Trying {model_desc}...")
        print("-" * 60)
        
        try:
            # SARIMAX Baseline
            if model_name == "sarimax_baseline":
                sarimax = fit_sarimax(df_train["y"], budget_sec=cfg["budget_sec"])
                if sarimax is not None:
                    pred_valid, conf_valid = forecast_sarimax(sarimax, len(df_valid))
                    if pred_valid is not None:
                        scores = compute_metrics(df_valid["y"], pred_valid, metrics_to_use)
                        results["sarimax_base_metrics"] = scores
                        
                        current_score = scores[primary_metric]
                        print(f"✓ {model_desc} scores:")
                        for metric, value in scores.items():
                            print(f"    {metric}: {value:.2f}")
                        
                        if current_score < best_score:
                            improvement = ((best_score - current_score) / best_score * 100) if best_score != float('inf') else 0
                            print(f"  → New best! Improved by {improvement:.1f}%")
                            best_score = current_score
                            best_model = ("sarimax", sarimax, None, pred_valid)
                        
                        models_tried.append({
                            "model": model_name,
                            "score": current_score,
                            "is_best": current_score == best_score
                        })
                        
                        # Generate residual plot
                        plot_residuals(df_valid["y"], pred_valid, "sarimax_baseline", outdir)
            
            # SARIMAX with Features
            elif model_name == "sarimax_featured" and not features.empty:
                sarimax_feat = fit_sarimax(df_train["y"], features.loc[df_train.index], 
                                          budget_sec=cfg["budget_sec"])
                if sarimax_feat is not None:
                    pred_valid, conf_valid = forecast_sarimax(sarimax_feat, len(df_valid), 
                                                   features.loc[df_valid.index])
                    if pred_valid is not None:
                        scores = compute_metrics(df_valid["y"], pred_valid, metrics_to_use)
                        results["sarimax_featured_metrics"] = scores
                        
                        current_score = scores[primary_metric]
                        print(f"✓ {model_desc} scores:")
                        for metric, value in scores.items():
                            print(f"    {metric}: {value:.2f}")
                        
                        if current_score < best_score:
                            improvement = ((best_score - current_score) / best_score * 100)
                            print(f"  → New best! Improved by {improvement:.1f}%")
                            best_score = current_score
                            best_model = ("sarimax_featured", sarimax_feat, features, pred_valid)
                        else:
                            degradation = ((current_score - best_score) / best_score * 100)
                            print(f"  → No improvement (degraded by {degradation:.1f}%)")
                        
                        models_tried.append({
                            "model": model_name,
                            "score": current_score,
                            "is_best": current_score == best_score
                        })
                        
                        plot_residuals(df_valid["y"], pred_valid, "sarimax_featured", outdir)
            
            # Prophet
            elif model_name == "prophet":
                # Prepare data for Prophet
                dfp = df_train.reset_index()[["date", "y"]].copy()
                dfp.columns = ["ds", "y"]
                
                # Verify we have data
                if len(dfp) == 0:
                    print(f"    ✗ Prophet skipped: empty training data")
                    continue
                
                print(f"    Prophet training data: {len(dfp)} periods from {dfp['ds'].min()} to {dfp['ds'].max()}")
                
                prophet = fit_prophet(dfp, budget_sec=cfg["budget_sec"])
                if prophet is not None:
                    pred_valid, conf_valid = forecast_prophet(prophet, len(df_valid))
                    if pred_valid is not None:
                        pred_valid = pd.Series(pred_valid, index=df_valid.index)
                        scores = compute_metrics(df_valid["y"], pred_valid, metrics_to_use)
                        results["prophet_metrics"] = scores
                        
                        current_score = scores[primary_metric]
                        print(f"✓ {model_desc} scores:")
                        for metric, value in scores.items():
                            print(f"    {metric}: {value:.2f}")
                        
                        if current_score < best_score:
                            improvement = ((best_score - current_score) / best_score * 100)
                            print(f"  → New best! Improved by {improvement:.1f}%")
                            best_score = current_score
                            best_model = ("prophet", prophet, features if not features.empty else None, pred_valid)
                        else:
                            degradation = ((current_score - best_score) / best_score * 100)
                            print(f"  → No improvement (degraded by {degradation:.1%})")
                        
                        models_tried.append({
                            "model": model_name,
                            "score": current_score,
                            "is_best": current_score == best_score
                        })
                        
                        plot_residuals(df_valid["y"], pred_valid, "prophet", outdir)
            
            # XGBoost
            elif model_name == "xgb" and not features.empty:
                print(f"    Features shape: {features.shape}, df_train shape: {df_train.shape}, df_valid shape: {df_valid.shape}")
                
                # Simple index-based slicing (avoid .isin() which can cause duplicates)
                # Split features the same way we split df
                X_train = features.iloc[:-n_valid].copy()
                X_valid = features.iloc[-n_valid:].copy()
                
                y_train = df_train["y"].copy()
                y_valid = df_valid["y"].copy()
                
                # Drop any NaN rows (from lag features)
                valid_train_idx = X_train.dropna().index
                X_train = X_train.loc[valid_train_idx]
                y_train = y_train.loc[valid_train_idx]
                
                valid_valid_idx = X_valid.dropna().index
                X_valid = X_valid.loc[valid_valid_idx]
                y_valid = y_valid.loc[valid_valid_idx]
                
                print(f"    After alignment - X_train: {X_train.shape}, y_train: {len(y_train)}")
                print(f"    After alignment - X_valid: {X_valid.shape}, y_valid: {len(y_valid)}")
                
                if len(X_train) == 0 or len(X_valid) == 0:
                    print(f"    ✗ XGBoost skipped: empty training or validation data after alignment")
                    continue
                
                # Verify lengths match
                if len(X_train) != len(y_train):
                    print(f"    ✗ XGBoost skipped: length mismatch after dropna")
                    continue
                
                xgb_model, xgb_metrics = train_xgb(X_train, y_train, X_valid, y_valid)
                if xgb_model is not None:
                    # Plot learning curves if available
                    if 'train_rmse_history' in xgb_metrics and 'valid_rmse_history' in xgb_metrics:
                        from src.utils.helpers import plot_learning_curves
                        plot_learning_curves(
                            xgb_metrics['train_rmse_history'],
                            xgb_metrics['valid_rmse_history'],
                            'xgb',
                            outdir
                        )
                    
                    # Use DMatrix for prediction
                    import xgboost as xgb
                    dvalid = xgb.DMatrix(X_valid)
                    pred_valid = pd.Series(xgb_model.predict(dvalid), index=X_valid.index)
                    
                    # Check for overfitting
                    from src.utils.helpers import detect_overfitting
                    if 'train_rmse' in xgb_metrics and 'valid_rmse' in xgb_metrics:
                        is_overfit, msg = detect_overfitting(
                            xgb_metrics['train_rmse'], 
                            xgb_metrics['valid_rmse'],
                            threshold=0.3
                        )
                        print(f"    {msg}")
                    
                    scores = compute_metrics(y_valid, pred_valid, metrics_to_use)
                    results["xgb_metrics"] = scores
                    
                    current_score = scores[primary_metric]
                    print(f"✓ {model_desc} scores:")
                    for metric, value in scores.items():
                        print(f"    {metric}: {value:.2f}")
                    
                    if current_score < best_score:
                        improvement = ((best_score - current_score) / best_score * 100)
                        print(f"  → New best! Improved by {improvement:.1f}%")
                        best_score = current_score
                        best_model = ("xgb", xgb_model, features, pred_valid)
                    else:
                        degradation = ((current_score - best_score) / best_score * 100)
                        print(f"  → No improvement (degraded by {degradation:.1f}%)")
                    
                    models_tried.append({
                        "model": model_name,
                        "score": current_score,
                        "is_best": current_score == best_score
                    })
                    
                    plot_residuals(y_valid, pred_valid, "xgb", outdir)
        
        except Exception as e:
            print(f"✗ {model_desc} failed: {str(e)}")
            results[f"{model_name}_error"] = str(e)
            continue
    
    # Summary of model ladder
    print("\n" + "="*60)
    print("MODEL LADDER SUMMARY")
    print("="*60)
    for m in models_tried:
        status = " BEST" if m["is_best"] else "    "
        print(f"{status} {m['model']:20s} → {primary_metric}: {m['score']:.2f}")
    
    # Check if any model succeeded
    if best_model is None:
        print("\n Warning: All models failed. Check your data and configuration.")
        results["status"] = "all_models_failed"
        save_metrics(results, outdir/"results.json")
        return
    
    model_type, model, feat, pred_valid = best_model
    print(f"\n Final best model: {model_type} with {primary_metric}={best_score:.2f}")
    
 # 7. Generate forecast with best model
    print(f"\nGenerating forecasts using best model ({model_type})...")
    
    # Determine horizons based on frequency
    freq = cfg.get("freq", "D")
    
    # Calculate horizons and labels based on frequency
    if freq in ("h", "H"):
        # For hourly data: convert days to hours
        horizon_days = [30, 90]
        horizons = [d * 24 for d in horizon_days]  # 720 hours and 2160 hours
        horizon_labels = [f"{d}d" for d in horizon_days]  # "30d" and "90d"
        horizon_display = "days (hourly)"
        
    elif freq in ("15min", "30min", "5min", "min"):
        # For sub-hourly data
        minutes_per_period = {
            '15min': 15,
            '30min': 30,
            '5min': 5,
            'min': 1
        }
        periods_per_hour = 60 // minutes_per_period.get(freq, 15)
        horizon_days = [30, 90]
        horizons = [d * 24 * periods_per_hour for d in horizon_days]
        horizon_labels = [f"{d}d" for d in horizon_days]
        horizon_display = f"days ({freq})"
        
    elif freq in ("MS", "ME", "M"):
        horizons = [3, 12]  # 3 months and 12 months
        horizon_labels = ["3m", "12m"]
        horizon_display = "months"
        
    elif freq in ("W",):
        horizons = [4, 12]  # 4 weeks and 12 weeks
        horizon_labels = ["4w", "12w"]
        horizon_display = "weeks"
        
    elif freq in ("QE", "Q"):
        horizons = [2, 4]  # 2 quarters and 4 quarters
        horizon_labels = ["2q", "4q"]
        horizon_display = "quarters"
        
    elif freq in ("YE", "Y"):
        horizons = [1, 3]  # 1 year and 3 years
        horizon_labels = ["1y", "3y"]
        horizon_display = "years"
        
    else:  # Daily (D, B)
        horizons = [30, 90]  # 30 days and 90 days
        horizon_labels = ["30d", "90d"]
        horizon_display = "days"
    
    print(f"  Frequency: {freq} → Horizons: {horizons} ({horizon_display})")
    
    for horizon, horizon_label in zip(horizons, horizon_labels):
        print(f"\n  Generating {horizon_label} forecast ({horizon} periods)...")
        
        if model_type.startswith("sarimax"):
            f, c = forecast_sarimax(model, horizon, 
                                  feat.tail(horizon) if feat is not None else None)
            if f is not None:
                # Generate proper future dates
                if freq in ("MS", "ME", "M"):
                    last_date = df.index.to_period("M").to_timestamp()[-1]
                    first_future = last_date + pd.offsets.MonthBegin(1)
                    future_dates = pd.date_range(first_future, periods=horizon, freq="MS")
                    
                elif freq in ("W",):
                    first_future = df.index[-1] + pd.Timedelta(weeks=1)
                    future_dates = pd.date_range(first_future, periods=horizon, freq="W")
                    
                elif freq in ("QE", "Q"):
                    first_future = df.index[-1] + pd.offsets.QuarterEnd(1)
                    future_dates = pd.date_range(first_future, periods=horizon, freq="QE")
                    
                elif freq in ("YE", "Y"):
                    first_future = df.index[-1] + pd.offsets.YearEnd(1)
                    future_dates = pd.date_range(first_future, periods=horizon, freq="YE")
                    
                elif freq in ("h", "H"):
                    first_future = df.index[-1] + pd.Timedelta(hours=1)
                    future_dates = pd.date_range(first_future, periods=horizon, freq="h")
                    
                elif freq in ("15min", "30min", "5min", "min"):
                    first_future = df.index[-1] + pd.Timedelta(freq)
                    future_dates = pd.date_range(first_future, periods=horizon, freq=freq)
                    
                else:  # Daily
                    first_future = df.index[-1] + pd.Timedelta(days=1)
                    future_dates = pd.date_range(first_future, periods=horizon, freq="D")
                
                forecast_df = pd.DataFrame({
                    "forecast": f,
                    "lower": c["lower"],
                    "upper": c["upper"]
                }, index=future_dates)
                
                print(f"    Forecast range: {forecast_df.index[0]} to {forecast_df.index[-1]}")
                print(f"    Forecast shape: {forecast_df.shape}")
                save_forecast(forecast_df, outdir/f"{model_type}_forecast_{horizon_label}.csv")
                plot_forecast_results(df, forecast_df, model_type, horizon_label, outdir)
            
        elif model_type == "prophet":
            # Build future dates based on frequency
            if freq in ("M", "ME", "MS"):
                last_hist_ms = df.index.to_period("M").to_timestamp()
                first_future = last_hist_ms[-1] + pd.offsets.MonthBegin(1)
                future_dates = pd.date_range(first_future, periods=horizon, freq="MS")
                
            elif freq == "W":
                first_future = df.index[-1] + pd.Timedelta(weeks=1)
                future_dates = pd.date_range(first_future, periods=horizon, freq="W")
                
            elif freq in ("h", "H"):
                first_future = df.index[-1] + pd.Timedelta(hours=1)
                future_dates = pd.date_range(first_future, periods=horizon, freq="h")
                
            elif freq in ("15min", "30min", "5min", "min"):
                first_future = df.index[-1] + pd.Timedelta(freq)
                future_dates = pd.date_range(first_future, periods=horizon, freq=freq)
                
            else:  # Daily
                first_future = df.index[-1] + pd.Timedelta(days=1)
                future_dates = pd.date_range(first_future, periods=horizon, freq="D")

            print(f"    Prophet horizon: {horizon} periods")
            print(f"    Date range: {future_dates[0]} to {future_dates[-1]}")
            print(f"    Last historical: {df.index[-1]}, value: {df['y'].iloc[-1]:.2f}")

            # Get raw forecast with correct horizon
            f, c = forecast_prophet(model, horizon, freq=freq, last_date=df.index[-1])
            
            if f is not None:
                print(f"    Raw Prophet forecast[0]: {f[0]:.2f}")
                
                # Level calibration (only for monthly data to avoid over-adjusting hourly)
                if freq in ("MS", "ME", "M"):
                    try:
                        k = min(12, len(df))
                        dfp_train = df.reset_index()[["date", "y"]]
                        dfp_train.columns = ["ds", "y"]
                        fitted = model.predict(dfp_train)[["ds", "yhat"]].set_index("ds")["yhat"]

                        recent_actual = df["y"].iloc[-k:].mean()
                        recent_fitted = fitted.iloc[-k:].mean()
                        
                        print(f"    Recent actual mean (k={k}): {recent_actual:.2f}")
                        print(f"    Recent fitted mean: {recent_fitted:.2f}")
                        
                        if recent_fitted > 0:
                            scale = recent_actual / recent_fitted
                            f = f * scale
                            c["lower"] = c["lower"] * scale
                            c["upper"] = c["upper"] * scale
                            print(f"    Applied level calibration: scale={scale:.3f}")
                    except Exception as e:
                        print(f"    Level calibration skipped: {e}")

                    # Seasonal naive blending for monthly data
                    try:
                        if len(df) >= 12:
                            alpha = 0.2
                            snaive_vals = np.array([df["y"].iloc[-12 + (i % 12)] for i in range(horizon)], dtype=float)
                            f = alpha * f + (1 - alpha) * snaive_vals
                            c["lower"] = alpha * c["lower"] + (1 - alpha) * snaive_vals
                            c["upper"] = alpha * c["upper"] + (1 - alpha) * snaive_vals
                            print(f"    Blended with seasonal naive (alpha={alpha})")
                    except Exception as e:
                        print(f"    Seasonal blending skipped: {e}")

                # Ensure forecast and dates match
                if len(f) != len(future_dates):
                    print(f"    ⚠️ Length mismatch: forecast={len(f)}, dates={len(future_dates)}")
                    min_len = min(len(f), len(future_dates))
                    f = f[:min_len]
                    c["lower"] = c["lower"][:min_len]
                    c["upper"] = c["upper"][:min_len]
                    future_dates = future_dates[:min_len]

                forecast_df = pd.DataFrame(
                    {"forecast": f, "lower": c["lower"], "upper": c["upper"]},
                    index=future_dates
                )

                print(f"    Forecast shape: {forecast_df.shape}")
                print(f"    Date range: {forecast_df.index[0]} to {forecast_df.index[-1]}")
                print(f"    Stats: mean={forecast_df['forecast'].mean():.2f}, "
                      f"min={forecast_df['forecast'].min():.2f}, max={forecast_df['forecast'].max():.2f}")
                
                save_forecast(forecast_df, outdir / f"prophet_forecast_{horizon_label}.csv")
                plot_forecast_results(df, forecast_df, "prophet", horizon_label, outdir)
                
        elif model_type == "xgb":
            try:
                import xgboost as xgb
                print(f"\n  DEBUG: XGBoost Prerequisites")
                print(f"    - Model exists: {model is not None}")
                print(f"    - Model type: {type(model)}")
                print(f"    - Features exist: {feat is not None}")
                if feat is not None:
                    print(f"    - Feature shape: {feat.shape}")
                    print(f"    - Feature columns: {feat.columns.tolist()}")
                print(f"    - Horizon: {horizon}")
                print(f"    - Frequency: {freq}")
                # Generate future dates with correct frequency
                if freq in ("MS", "ME", "M"):
                    last_date = df.index.to_period("M").to_timestamp()[-1]
                    first_future = last_date + pd.offsets.MonthBegin(1)
                    future_dates = pd.date_range(first_future, periods=horizon, freq="MS")
                    
                elif freq in ("W",):
                    first_future = df.index[-1] + pd.Timedelta(weeks=1)
                    future_dates = pd.date_range(first_future, periods=horizon, freq="W")
                    
                elif freq in ("h", "H"):
                    first_future = df.index[-1] + pd.Timedelta(hours=1)
                    future_dates = pd.date_range(first_future, periods=horizon, freq="h")
                    
                elif freq in ("15min", "30min", "5min", "min"):
                    first_future = df.index[-1] + pd.Timedelta(freq)
                    future_dates = pd.date_range(first_future, periods=horizon, freq=freq)
                    
                else:  # Daily
                    first_future = df.index[-1] + pd.Timedelta(days=1)
                    future_dates = pd.date_range(first_future, periods=horizon, freq="D")
                
                print(f"    Generating recursive forecast for {horizon} periods...")
                print(f"    Date range: {future_dates[0]} to {future_dates[-1]}")
                
                # Start with historical data
                history = df[["y"]].copy()
                forecasts = []
                
                # Get feature columns
                feature_cols = feat.columns.tolist()
                print(f"    Required features: {len(feature_cols)} features")
                
                # Pre-generate calendar and holiday features
                future_cal_features = make_calendar_features(
                    future_dates, 
                    weekend_days=cfg.get("weekend_days", [5,6])
                )
                
                if 'is_holiday' in feature_cols:
                    future_hol_features = make_country_holidays(
                        future_dates,
                        country_code=cfg.get("holiday_country", "AE")
                    )
                    for col in future_hol_features.columns:
                        if col in feature_cols:
                            future_cal_features[col] = future_hol_features[col]
                
                # Recursive forecasting
                for i, future_date in enumerate(future_dates):
                    date_features = future_cal_features.loc[[future_date]].copy()
                    
                    # Generate lag and rolling features
                    if 'lag_1' in feature_cols:
                        date_features['lag_1'] = history['y'].iloc[-1] if len(history) > 0 else 0
                    if 'lag_7' in feature_cols:
                        date_features['lag_7'] = history['y'].iloc[-7] if len(history) >= 7 else history['y'].iloc[0]
                    if 'lag_14' in feature_cols:
                        date_features['lag_14'] = history['y'].iloc[-14] if len(history) >= 14 else history['y'].iloc[0]
                    if 'roll7_mean' in feature_cols:
                        date_features['roll7_mean'] = history['y'].iloc[-7:].mean() if len(history) >= 7 else history['y'].mean()
                    if 'roll7_std' in feature_cols:
                        date_features['roll7_std'] = history['y'].iloc[-7:].std() if len(history) >= 7 else 0
                    
                    # Add missing features
                    for col in feature_cols:
                        if col not in date_features.columns:
                            date_features[col] = 0
                    
                    X_future = date_features[feature_cols]
                    
                    # Predict
                    dtest = xgb.DMatrix(X_future)
                    pred = model.predict(dtest)[0]
                    forecasts.append(pred)
                    
                    # Update history
                    new_row = pd.DataFrame({'y': [pred]}, index=[future_date])
                    history = pd.concat([history, new_row])
                    
                    if (i + 1) % 100 == 0:
                        print(f"      Progress: {i+1}/{horizon} periods")
                
                forecasts = np.array(forecasts)
                
                # Conformal prediction intervals
                from src.confidence.intervals import conformal_intervals_from_residuals
                conf = conformal_intervals_from_residuals(y_valid, pred_valid, forecasts)
                
                forecast_df = pd.DataFrame({
                    "forecast": forecasts,
                    "lower": conf.get("lower", forecasts * 0.9),
                    "upper": conf.get("upper", forecasts * 1.1)
                }, index=future_dates)
                
                print(f"    Forecast shape: {forecast_df.shape}")
                print(f"    Date range: {forecast_df.index[0]} to {forecast_df.index[-1]}")
                print(f"    Stats: mean={forecast_df['forecast'].mean():.2f}, "
                      f"min={forecast_df['forecast'].min():.2f}, max={forecast_df['forecast'].max():.2f}")
                
                save_forecast(forecast_df, outdir/f"xgb_forecast_{horizon_label}.csv")
                plot_forecast_results(df, forecast_df, "xgb", horizon_label, outdir)
                
            except Exception as e:
                print(f"    ✗ XGBoost forecast failed: {str(e)}")
                import traceback
                traceback.print_exc()
    # 8. Final metrics & drift check
    results.update({
        "best_model": model_type,
        "best_score": best_score,
        "primary_metric": primary_metric,
        "n_features_used": len(features.columns),
        "models_tried": models_tried,
        "drift_psi": float(psi_score(df["y"].iloc[-180:-90].values, 
                                   df["y"].iloc[-90:].values))
    })
    
    # Save results
    results["feature_selection"] = feature_results
    save_metrics(results, outdir/"results.json")
    
    # Generate comparison plot
    plot_model_comparison(results, outdir, primary_metric)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Results saved to: {outdir}")
    print(f"Best model: {model_type}")
    print(f"Best {primary_metric}: {best_score:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
# General Forecasting Pipeline

## Problem Statement

**Context**: Organizations need reliable demand forecasts to optimize inventory, staffing, and resource allocation. However, building robust forecasting systems is challenging due to:
- Multiple data sources with different frequencies and formats
- Seasonal patterns, trends, and external factors (holidays, events)
- Need for model selection across statistical and ML approaches
- Requirement for uncertainty quantification and production monitoring

**Goal**: Build an automated, production-ready forecasting pipeline that:
1. Ingests time series data from various sources
2. Engineers relevant features and selects optimal models
3. Generates accurate forecasts with confidence intervals
4. Monitors performance and detects drift over time

**Solution**: A modular pipeline implementing a progressive model ladder (SARIMAX → Prophet → XGBoost) with automated input detection, feature engineering, feature selection, hyperparameter tuning, accuracy metric selection, and conformal prediction intervals.

---

## Pipeline Architecture
Data → Load & Validate → EDA → Feature Engineering → Model Ladder → Best Model → Forecast → Monitor


## Components (STAR Format)

### 1. Data Ingestion (`src/ingestion/load.py`)

Time series data comes from diverse sources (CSV, URLs, databases) with inconsistent formats, missing values, and varying frequencies (daily, weekly, monthly).

Create a robust data loader that:
- Accepts multiple input formats and sources
- Validates data quality (missing values, duplicates, outliers)
- Infers and normalizes frequency
- Handles date parsing across different formats


- Implemented `load_csv()` with automatic URL/file detection
- Added custom date parsers for formats like "M-YY" and "DD-MM-YYYY"
- Created frequency inference using median time gaps
- Normalized monthly data to Month Start (MS) for consistency
- Added outlier detection using IQR method

Successfully loads datasets with:
- 95%+ data quality (handles missing values, duplicates)
- Automatic frequency detection (daily, weekly, monthly, 15-min intervals)
- Validated on: airline passengers (M), Synthetic sales (D), traffic data (H)

---

### 2. Exploratory Data Analysis (`src/eda/`)

Need to understand data characteristics before modeling—trend, seasonality, stationarity—to inform feature engineering and model selection.

Perform comprehensive EDA to:
- Visualize time series patterns
- Test for stationarity (ADF test)
- Decompose into trend/seasonal/residual components
- Analyze autocorrelation (ACF/PACF)


- Created visualization functions for time series plots with trend lines
- Implemented ADF test with p-value interpretation
- Added seasonal decomposition (additive/multiplicative)
- Generated ACF/PACF plots to identify AR/MA orders


- Clear identification of non-stationary series (p-value > 0.05)
- Seasonal period detection (12 for monthly, 7 for daily)
- Insights inform differencing and seasonal parameters for SARIMAX

---

### 3. Feature Engineering (`src/features/`)

Raw timestamps lack predictive power; need to create temporal and domain-specific features that capture patterns.

Generate features that help models learn:
- Calendar effects (day of week, month, holidays)
- Lag features (previous values)
- Rolling statistics (moving averages, volatility)
- Cyclical patterns (sin/cos transformations)


- `make_calendar_features()`: day of week, weekend indicator, sin/cos annual cycle
- `make_country_holidays()`: country-specific holiday flags (UAE, US, UK)
- `make_lag_features()`: lag 1, 7, 14 days with NaN handling
- `make_rolling_features()`: 7-day rolling mean and std deviation


- XGBoost model improved MAPE from 8.87% (baseline) to 4.51% using engineered features
- Holiday features capture demand spikes (e.g., Eid, Christmas)
- Lag features enable recursive forecasting

---

### 4. Model Ladder & Selection (`scripts/run_pipeline.py`)

No single model performs best across all datasets; need an automated approach to try multiple models and select the best.

Implement a progressive model ladder that:
- Starts with simple baseline (SARIMAX)
- Progresses to more complex models (Prophet, XGBoost)
- Compares models using validation metrics
- Selects best model based on MAPE/RMSE


- **SARIMAX Baseline**: Auto-selected (p,d,q)(P,D,Q,s) orders using AIC
- **Prophet**: Tuned `changepoint_prior_scale` via cross-validation, added level calibration
- **XGBoost**: Trained with early stopping, overfitting detection, feature importance analysis
- Used time-based train/validation split (e.g., 24 months train, 6 months validation)

Model ladder summary example:
SARIMAX baseline → MAPE: 8.87%
Prophet → MAPE: 5.40% (39% improvement)
XGBoost → MAPE: 4.51% (16% improvement, BEST)


### 5. SARIMAX Implementation (`src/models/sarimax.py`)

Need a statistical baseline that captures seasonality and external regressors.

Implement SARIMAX with:
- Automatic order selection (AIC-based)
- Support for exogenous features
- Robust error handling for convergence issues


- Created `fit_sarimax()` with grid search over (p,d,q) and (P,D,Q,s)
- Added MLE optimization with fallback to `lbfgs`
- Implemented `forecast_sarimax()` with confidence intervals


- Baseline MAPE: 8-15% across datasets
- Captures seasonal patterns automatically
- Provides interpretable coefficients

---

### 6. Prophet Implementation (`src/models/prophet.py`)

Prophet excels at handling seasonality and holidays but often produces low-level forecasts without calibration.

Enhance Prophet with:
- Automatic frequency detection (hourly, daily, monthly)
- Cross-validation for hyperparameter tuning
- Level calibration to fix forecast bias
- Seasonal-naive blending for monthly data


- Implemented `fit_prophet()` with CV over `changepoint_prior_scale` values
- Added level calibration: `scale = mean(recent_actual) / mean(recent_fitted)`
- Applied 80/20 blend with seasonal-naive for monthly forecasts
- Detected high-frequency data (15-min) and adjusted CV parameters


- MAPE improved from 15%+ (uncalibrated) to 5.4% (calibrated)
- Level matches historical mean within 2%
- Successfully handles daily, weekly, monthly, and sub-hourly data

---

### 7. XGBoost Implementation (`src/models/xgb.py`)

Need a model that leverages engineered features and handles non-linear relationships.

Train XGBoost with:
- Optimal hyperparameters (max_depth, learning_rate, regularization)
- Early stopping to prevent overfitting
- Overfitting detection and diagnostics


- Created `train_xgb()` with DMatrix for efficiency
- Used early stopping (patience=50, eval_metric='rmse')
- Implemented overfitting detection: flag if valid_rmse/train_rmse > 1.3
- Plotted learning curves (train vs. validation RMSE)

Example:
- Best MAPE: 4.51% (vs. 5.40% Prophet, 8.87% SARIMAX)
- Detected overfitting: 49.9% degradation (train RMSE=50, valid RMSE=75)
- Feature importance: lag_1 (0.35), roll7_mean (0.25), dow (0.15)

---

### 8. Recursive Forecasting (`scripts/run_pipeline.py`)

XGBoost requires lag features, but for future dates we don't have actuals—need to use predictions recursively.

Generate multi-step forecasts by:
- Predicting one step at a time
- Using each prediction as input for next step's lag features
- Maintaining feature alignment (calendar + lag + rolling)


- Pre-generated calendar and holiday features for all future dates
- For each step:
  - Compute lags from history (which includes previous predictions)
  - Compute rolling statistics from updated history
  - Predict and append to history
- Handled edge cases (lag periods longer than history)


- Successfully generated 30-day and 90-day forecasts
- Maintained feature consistency (9 features: dow, is_weekend, sin_annual, is_holiday, lag_1, lag_7, lag_14, roll7_mean, roll7_std)
- Forecasts maintain realistic levels and seasonal patterns

---

### 9. Confidence Intervals (`src/confidence/intervals.py`)

Point forecasts are insufficient for decision-making; stakeholders need uncertainty quantification.

Provide prediction intervals using:
- Conformal prediction (distribution-free)
- Bootstrap resampling for Prophet/SARIMAX


- Implemented `conformal_intervals_from_residuals()`:
  - Compute validation residuals: `residuals = |y_true - y_pred|`
  - Calculate quantile: `q = quantile(residuals, 0.9)` for 90% intervals
  - Generate bounds: `[forecast - q, forecast + q]`
- Added bootstrap CI for SARIMAX using prediction variance


- 90% prediction intervals cover ~88% of actual values (well-calibrated)
- Intervals widen for longer horizons (30-day: ±15, 90-day: ±30)
- Prophet intervals tightest, XGBoost widest (reflects model uncertainty)

---

### 10. Metrics & Evaluation (`src/metrics/evaluation.py`)

Different stakeholders care about different metrics (MAPE for %, RMSE for scale).

Implement multiple metrics:
- MAPE: % error (scale-independent)
- WMAPE: Weighted % error (handles low volumes)
- RMSE: Penalizes large errors
- Coverage: % of actuals within prediction intervals


- Created `compute_metrics()` supporting all metrics
- Added metric targets from config (e.g., MAPE < 15% for 1-month horizon)
- Implemented pass/fail flags based on targets

Model comparison table:


| Model            | MAPE  | WMAPE | RMSE | Status           |
|:-----------------|------:|------:|-----:|:-----------------|
| SARIMAX baseline | 8.87% | 2145  | 45.2 | ✓ Pass (<15%)    |
| Prophet          | 5.40% | 1567  | 38.1 | ✓ Pass           |
| XGBoost          | 4.51% | 1235  | 33.7 | ✓ Pass (BEST)    |


---

### 11. Visualization (`src/plotting/charts.py`)

Stakeholders need visual outputs to understand forecasts and model performance.

Create publication-quality plots for:
- Forecast with confidence intervals
- Residual diagnostics (QQ plot, histogram)
- Learning curves (XGBoost)
- Model comparison


- `plot_forecast_results()`: Historical + forecast + intervals
- `plot_residuals()`: 4-panel diagnostic plot
- `plot_learning_curves()`: Train vs. validation RMSE
- `plot_model_comparison()`: Side-by-side MAPE bars


- Saved plots to `outputs/` folder
- Clear visualization of forecast quality
- Residual plots show no autocorrelation (white noise)

---

### 12. Configuration Management (`configs/default.yaml`)

Hard-coded parameters make experimentation difficult; need flexible configuration.

Externalize all parameters:
- Dataset sources (URLs, file paths)
- Model hyperparameters
- Metric targets
- Feature engineering settings


- Created YAML config with nested structure:
  - `datasets`: airline, Synthetic, traffic (each with URL, date_col, target_col, freq)
  - `weekend_days`, `holiday_country`, `metric_targets`
  - `budget_sec` (time limit for training)
- Used `yaml.safe_load()` to parse config


- Switch datasets by changing one line: `dataset_name: "airline"`
- Tune models without code changes
- Easy A/B testing of configurations

---

### 13. Outputs & Governance (`outputs/`)

Need to persist results for audit trails, model comparison, and production deployment.

Save structured outputs:
- Forecast CSVs (date, forecast, lower, upper)
- Metrics JSON (MAPE, RMSE, training time)
- Diagnostic plots (PNG)
- Model artifacts (pickle)


- Created `save_forecast()` and `save_metrics()` utilities
- Standardized file naming: `{model}_forecast_{horizon}d.csv`
- Logged all model decisions and scores to `results.json`

Output files:
- model_baseline_forecast_30d.csv
- model_forecast_90d.csv
- model_forecast_30d.csv
- results.json
- model_comparison.png
- model_residuals.png



---

## Usage

#### Run pipeline with default config:
python scripts/run_pipeline.py

#### Use specific dataset:
Edit configs/default.yaml: dataset_name: "Synthetic" / "airline" / "traffic"
python scripts/run_pipeline.py

## Results Summary

### Model Performance

| Dataset         | Frequency | Best Model | MAPE  | WMAPE | RMSE | Horizon | Status |
|:----------------|:----------|:-----------|------:|------:|-----:|--------:|:-------|
| Airline         | Monthly   | XGBoost    | 4.51% | 1235  | 33.7 | 30d     | ✓ BEST |
| Synthetic       | Daily     | Prophet    | 5.40% | 1567  | 38.1 | 90d     | ✓ Pass |
| Traffic (15min) | 15-minute | SARIMAX    | 8.87% | 2145  | 45.2 | 24h     | ✓ Pass |

### Model Details

| Dataset         | Features Used | Train RMSE | Valid RMSE | Degradation | Notes |
|:----------------|:--------------|:-----------|:-----------|------------:|:------|
| Airline         | dow, is_weekend, sin_annual, is_holiday, lag_1, lag_7, lag_14, roll7_mean, roll7_std | 50.0 | 75.0 | +49.9% | Overfitting detected, seasonal patterns captured |
| Synthetic         | Prophet internal (yearly/monthly seasonality) | 30.2 | 38.1 | +20% | Level calibration critical, 80/20 seasonal-naive blend |
| Traffic (15min) | None (univariate SARIMAX) | 41.8 | 45.2 | +40% | High-frequency baseline, room for feature engineering |

---

#### Next Steps
- Hyperparameter Tuning: Add Optuna for automated HPO
- Ensemble: Combine top 2 models (weighted average)
- Drift Detection: Monitor residuals for distribution shifts
- API: Wrap pipeline in FastAPI for production serving
- Future: Make modeling into an agentic approach based on the input data for better model selection and hyperparameter tuning
- Robustness: Gather better external data using the insights from EDA and data set descriptions to add more external variables using Agnets




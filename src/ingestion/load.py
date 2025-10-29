import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Dict, Any, Tuple
import requests
from requests.exceptions import Timeout

def is_url(path: str) -> bool:
    """Check if the path is a URL."""
    try:
        result = urlparse(path)
        print(f"Parsed URL: {result}")
        return all([result.scheme, result.netloc])
    except:
        return False

def infer_frequency(index: pd.DatetimeIndex) -> Tuple[str, pd.Timedelta]:
    """
    Infer the frequency of a datetime index.
    
    Returns:
        Tuple of (frequency_string, median_gap)
    """
    if len(index) < 2:
        return "D", pd.Timedelta(days=1)
    
    # Calculate gaps between consecutive dates
    gaps = index.to_series().diff().dropna()
    median_gap = gaps.median()
    
    # Determine frequency based on median gap
    if median_gap <= pd.Timedelta(hours=1):
        return "H", median_gap
    elif median_gap <= pd.Timedelta(days=1):
        return "D", median_gap
    elif median_gap <= pd.Timedelta(days=7):
        return "W", median_gap
    elif median_gap <= pd.Timedelta(days=31):
        return "ME", median_gap  # Month End
    elif median_gap <= pd.Timedelta(days=92):
        return "QE", median_gap  # Quarter End
    else:
        return "YE", median_gap  # Year End

def combine_date_columns(df: pd.DataFrame, date_cols: list[str], combined_col: str = "date") -> pd.DataFrame:
    """Combine multiple date-related columns into a single datetime column."""
    if len(date_cols) == 2 and date_cols[0].lower() in ["date", "dt"] and date_cols[1].lower() == "time":
        df[combined_col] = pd.to_datetime(df[date_cols[0]] + " " + df[date_cols[1]])
    else:
        df[combined_col] = pd.to_datetime(
            {
                'year': df[date_cols[0]] if 'year' in date_cols[0].lower() else None,
                'month': df[date_cols[1]] if len(date_cols) > 1 else None,
                'day': df[date_cols[2]] if len(date_cols) > 2 else None,
                'hour': df[date_cols[3]] if len(date_cols) > 3 else None
            }
        )
    return df

def validate_loaded_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    """Validate loaded DataFrame meets requirements."""
    min_rows = cfg.get("min_rows", 30)
    if len(df) < min_rows:
        raise ValueError(f"Dataset too small: {len(df)} rows < minimum {min_rows}")
    
    y = df.get("y")
    if y is None:
        raise ValueError("Target column 'y' not found")
    
    if y.min() < 0 and not cfg.get("allow_negative", False):
        raise ValueError(f"Negative values found in target: min={y.min()}")
    
    missing_pct = y.isna().mean()
    max_missing = cfg.get("max_missing_pct", 0.2)
    if missing_pct > max_missing:
        raise ValueError(f"Too many missing values: {missing_pct:.1%} > {max_missing:.1%}")
    
    if cfg.get("check_outliers", True):
        q1, q3 = y.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((y < (q1 - 1.5 * iqr)) | (y > (q3 + 1.5 * iqr))).mean()
        if outliers > cfg.get("max_outlier_pct", 0.1):
            import warnings
            warnings.warn(f"High outlier percentage: {outliers:.1%}")

def load_csv(cfg: dict) -> pd.DataFrame:
    """
    Load CSV from URL or local file with standardized processing.
    """
    # Get dataset-specific config
    dataset_name = cfg.get("dataset_name")
    if dataset_name and dataset_name in cfg.get("datasets", {}):
        dataset_cfg = cfg["datasets"][dataset_name]
        print(f"Using dataset configuration: {dataset_name}")
        
        url = dataset_cfg.get("url")
        if url:
            cfg["file_candidates"] = [url] + cfg.get("file_candidates", [])
        
        for key in ["date_col", "target_col", "freq", "compression", "sep", "na_values", 
                   "date_cols", "combined_date_col", "group_by", "hierarchical", "id_cols"]:
            if key in dataset_cfg:
                cfg[key] = dataset_cfg[key]
                print(f"  {key}: {dataset_cfg[key]}")

    required_keys = ["file_candidates"]
    missing_keys = [k for k in required_keys if k not in cfg]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    timeout_sec = cfg.get("timeout_sec", 30)
    path = None
    df_raw = None
    
    print(f"\nTrying {len(cfg['file_candidates'])} candidate sources...")
    
    for cand in cfg.get("file_candidates", []):
        try:
            if is_url(cand):
                print(f"Attempting to load from URL: {cand}")
                response = requests.get(cand, timeout=timeout_sec)
                response.raise_for_status()
                
                read_kwargs = {
                    "parse_dates": cfg.get("parse_dates", [cfg.get("date_col", "date")]),
                    "compression": cfg.get("compression", "infer"),
                    "sep": cfg.get("sep", ","),
                    "na_values": cfg.get("na_values", ["NA", ""]),
                    "nrows": cfg.get("max_rows", None),
                }
                
                df_raw = pd.read_csv(cand, **read_kwargs)
                path = cand
                print(f"✓ Successfully loaded data from URL: {cand}")
                break
            else:
                print(f"Attempting to load local file: {cand}")
                p = Path(cand)
                if p.exists():
                    read_kwargs = {
                        "parse_dates": cfg.get("date_cols", [cfg.get("date_col", "date")]),
                        "compression": cfg.get("compression", "infer"),
                        "sep": cfg.get("sep", ","),
                        "na_values": cfg.get("na_values", ["NA", ""]),
                    }
                    df_raw = pd.read_csv(p, **read_kwargs)
                    path = p
                    print(f"✓ Successfully loaded local file: {p}")
                    break
                else:
                    print(f"✗ Local file does not exist: {p}")
                    continue

        except Timeout:
            print(f"✗ Timeout loading from {cand} after {timeout_sec}s")
            continue
        except requests.HTTPError as e:
            print(f"✗ HTTP error loading {cand}: {e}")
            continue
        except Exception as e:
            print(f"✗ Failed to load {cand}: {str(e)}")
            continue
    
    if df_raw is None:
        raise FileNotFoundError(f"No valid data source found. Tried {len(cfg['file_candidates'])} candidates.")

    print(f"\nLoaded data from: {path}")

    max_mb = cfg.get("max_size_mb", 1000)
    size_mb = df_raw.memory_usage(deep=True).sum() / 1e6
    if size_mb > max_mb:
        raise ValueError(f"Dataset too large: {size_mb:.1f}MB > {max_mb}MB")
    
    # Handle multiple date columns
    if cfg.get("date_cols") and len(cfg.get("date_cols", [])) > 1:
        df_raw = combine_date_columns(
            df_raw, 
            cfg["date_cols"], 
            cfg.get("combined_date_col", "date")
        )
        cfg["date_col"] = cfg.get("combined_date_col", "date")

    # Handle hierarchical/multi-series data
    if cfg.get("hierarchical"):
        id_cols = cfg.get("id_cols", [])
        if not id_cols:
            raise ValueError("id_cols must be specified for hierarchical data")
        df_raw['series_id'] = df_raw[id_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        first_series = df_raw['series_id'].iloc[0]
        print(f"Processing hierarchical data - using series: {first_series}")
        df_raw = df_raw[df_raw['series_id'] == first_series]
    
    if cfg.get("group_by"):
        group_col = cfg.get("group_by")
        first_group = df_raw[group_col].iloc[0]
        print(f"Processing multi-series data - using group: {first_group}")
        df_raw = df_raw[df_raw[group_col] == first_group]

    # Rename target column
    print("\nRaw data preview:")
    print(df_raw.head(3))
    target_col = cfg.get("target_col", "value")
    if target_col in df_raw.columns and target_col != "y":
        df_raw = df_raw.rename(columns={target_col: "y"})
    elif target_col not in df_raw.columns and "y" not in df_raw.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {df_raw.columns.tolist()}")

    # Set index and sort
    date_col = cfg.get("date_col", "date")
    
    # Ensure date column is datetime
    if date_col in df_raw.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_raw[date_col]):
            print(f"\nConverting {date_col} to datetime...")
            try:
                # First attempt: standard parsing with dayfirst for DD-MM-YYYY
                df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True)
                print(f"  ✓ Parsed with standard datetime conversion (dayfirst=True)")
            except Exception as e:
                print(f"  Standard conversion failed, trying custom formats...")
                
                # Check if it's the shampoo format (e.g., "1-01" = Jan 2001)
                sample = str(df_raw[date_col].iloc[0])
                if '-' in sample and len(sample.split('-')) == 2:
                    parts = sample.split('-')
                    if len(parts[0]) <= 2 and len(parts[1]) == 2:
                        # Format: M-YY or MM-YY
                        print(f"  Detected M-YY format (e.g., '1-01' = Jan 2001)")
                        def parse_month_year(s):
                            month, year = s.split('-')
                            # Assume 2000s for years 00-99
                            year = int(year) + 2000 if int(year) < 100 else int(year)
                            return pd.Timestamp(year=year, month=int(month), day=1)
                        
                        df_raw[date_col] = df_raw[date_col].apply(parse_month_year)
                        print(f"  ✓ Parsed with custom M-YY format")
                    else:
                        # Try standard date formats with dayfirst
                        for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y"]:
                            try:
                                df_raw[date_col] = pd.to_datetime(df_raw[date_col], format=fmt)
                                print(f"  ✓ Parsed with format: {fmt}")
                                break
                            except:
                                continue
                        else:
                            raise ValueError(f"Could not parse date column '{date_col}': {e}")
                else:
                    # Try standard date formats
                    for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y"]:
                        try:
                            df_raw[date_col] = pd.to_datetime(df_raw[date_col], format=fmt)
                            print(f"  ✓ Parsed with format: {fmt}")
                            break
                        except:
                            continue
                    else:
                        raise ValueError(f"Could not parse date column '{date_col}': {e}")
    
    # Combine date and time columns if both exist
    time_col = cfg.get("time_col", "Time")
    if time_col in df_raw.columns and date_col in df_raw.columns:
        print(f"\nCombining {date_col} and {time_col} into datetime index...")
        try:
            # Parse time column
            if not pd.api.types.is_datetime64_any_dtype(df_raw[time_col]):
                # Convert time string to timedelta
                df_raw[time_col] = pd.to_timedelta(df_raw[time_col])
            
            # Combine date + time
            df_raw[date_col] = pd.to_datetime(df_raw[date_col]) + df_raw[time_col]
            print(f"  ✓ Combined into full datetime")
            
            # Drop the separate time column
            df_raw = df_raw.drop(columns=[time_col])
        except Exception as e:
            print(f"  ⚠️ Could not combine date and time: {e}")
            print(f"  Continuing with date column only...")
    
    df = df_raw.sort_values(date_col).set_index(date_col)
    
    # Verify index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Index is not DatetimeIndex after setting. Got: {type(df.index)}")
    
    if "y" not in df.columns:
        raise ValueError("Column 'y' not found after renaming")
    
    print(f"\nBefore processing - y column stats:")
    print(f"  Count: {df['y'].count()}")
    print(f"  Mean: {df['y'].mean():.2f}")
    print(f"  Min: {df['y'].min():.2f}, Max: {df['y'].max():.2f}")
    print(f"  Index type: {type(df.index)}")
    print(f"  First date: {df.index[0]}, Last date: {df.index[-1]}")
    
    # Infer frequency from the data
    inferred_freq, median_gap = infer_frequency(df.index)
    config_freq = cfg.get("freq", "D")
    
    print(f"\nFrequency detection:")
    print(f"  Config frequency: {config_freq}")
    print(f"  Inferred frequency: {inferred_freq}")
    print(f"  Median gap: {median_gap}")
    
    # Use inferred frequency if not specified or if it differs significantly
    freq = inferred_freq if config_freq == "D" or config_freq != inferred_freq else config_freq
    cfg["freq"] = freq  # Update config with actual frequency
    
    # Check for gaps in the time series
    expected_periods = len(df)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    gaps_count = len(full_idx) - len(df)
    
    print(f"\nGap analysis:")
    print(f"  Current periods: {len(df)}")
    print(f"  Expected periods (freq={freq}): {len(full_idx)}")
    print(f"  Missing periods: {gaps_count}")
    
    # Only fill gaps if reasonable (< 10% missing)
    if gaps_count > 0:
        gap_pct = gaps_count / len(full_idx)
        if gap_pct < 0.1:
            print(f"  Filling {gaps_count} gaps ({gap_pct:.1%})...")
            original_idx = df.index.copy()
            df = df.reindex(full_idx)
            df["imputed_flag"] = (~df.index.isin(original_idx)).astype(int)
        else:
            print(f"  Too many gaps ({gap_pct:.1%}), keeping original frequency")
            df["imputed_flag"] = 0
    else:
        print("  No gaps detected")
        df["imputed_flag"] = 0
    
    df.index.name = "date"

    # Normalize monthly index to Month Start for consistency across pipeline
    if freq in ("M", "ME", "MS"):
        df.index = df.index.to_period("M").to_timestamp()  # Defaults to start
        df = df.sort_index()
        cfg["freq"] = "MS"
        freq = "MS"
        print(f"  Normalized monthly index to Month Start (MS)")    
    
    # Handle missing values
    if df["y"].isna().any():
        missing_before = df["y"].isna().sum()
        y = df["y"].copy()
        
        # Frequency-aware lookback
        lookback_map = {"H": 24, "D": 7, "W": 4, "ME": 12, "QE": 4, "YE": 1}
        lookback = lookback_map.get(freq, 1)
        
        for i in range(len(y)):
            if pd.isna(y.iat[i]) and i - lookback >= 0:
                y.iat[i] = y.iat[i - lookback]
        
        y = y.ffill().bfill()
        df["y"] = y
        
        missing_after = df["y"].isna().sum()
        print(f"\nImputation: {missing_before - missing_after} values filled")
    
    print(f"\nFinal dataset:")
    print(f"  Shape: {df.shape}")
    print(f"  Frequency: {freq}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Missing values: {df['y'].isna().sum()} ({df['y'].isna().mean():.1%})")
    if df['y'].notna().any():
        print(f"  Value range: {df['y'].min():.2f} to {df['y'].max():.2f}")
    
    # Validate
    validate_loaded_data(df, cfg)
    
    return df
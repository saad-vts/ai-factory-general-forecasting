import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any

def ensure_outdir(p: Path) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        p: Path object to create
    
    Returns:
        Path object that was created/existed
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_forecast(df_forecast: pd.DataFrame, path: str) -> str:
    """
    Save forecast DataFrame to CSV.
    
    Args:
        df_forecast: DataFrame containing forecasts
        path: Path to save CSV file
        
    Returns:
        Absolute path of saved file
    """
    p = ensure_outdir(Path(path).parent)
    df_forecast.to_csv(path)
    return str(Path(path).resolve())

def save_metrics(metrics: Dict[str, Any], path: str) -> str:
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Dictionary of metrics to save
        path: Path to save JSON file
        
    Returns:
        Absolute path of saved file
    """
    p = ensure_outdir(Path(path).parent)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    return str(Path(path).resolve())
import pandas as pd

def validate_cfg(cfg: dict):
    """
    Basic configuration validation. Paste into: src/ingestion/validators.py
    """
    assert isinstance(cfg, dict), "cfg must be a dict"
    assert "file_candidates" in cfg, "cfg['file_candidates'] missing"
    assert "date_col" in cfg, "cfg['date_col'] missing"
    assert "target_col" in cfg, "cfg['target_col'] missing"

def validate_df(df: pd.DataFrame):
    """
    Ensure df has index of datetime and a column 'y'. Paste into: src/ingestion/validators.py
    """
    assert df.index.dtype.kind in ("M",), "df index must be datetime"
    assert "y" in df.columns, "df must contain target column 'y'"
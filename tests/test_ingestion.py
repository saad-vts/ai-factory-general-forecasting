import pytest
import pandas as pd
from src.ingestion.load import load_csv
from src.ingestion.validators import validate_cfg, validate_df

def test_validate_cfg():
    valid_cfg = {
        'file_candidates': ['test.csv'],
        'date_col': 'date',
        'target_col': 'orders'
    }
    validate_cfg(valid_cfg)  # should not raise

    with pytest.raises(AssertionError):
        validate_cfg({})  # empty config should fail

def test_validate_df(sample_df):
    validate_df(sample_df)  # should not raise

    bad_df = pd.DataFrame({'wrong_col': [1,2,3]})
    with pytest.raises(AssertionError):
        validate_df(bad_df)
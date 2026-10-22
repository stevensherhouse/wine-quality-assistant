import json
from pathlib import Path
import joblib
import pandas as pd
import pytest
from src.preprocess import FEATURES

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='session')
def trained():
    directory = ROOT / 'models'
    assert (directory / 'best_model.joblib').exists(), 'First run: python -m src.train --download'
    return (joblib.load(directory / 'best_model.joblib'),
            pd.read_csv(directory / 'test_sample.csv'),
            json.loads((directory / 'metadata.json').read_text()))

@pytest.fixture
def measurements():
    return dict(zip(FEATURES, [7.4, .70, 0., 1.9, .076, 11., 34., .9978, 3.51, .56, 9.4, 'red']))

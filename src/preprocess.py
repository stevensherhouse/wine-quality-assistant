"""Shared feature contract and training-only preprocessing."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'pH', 'sulphates', 'alcohol',
]
FEATURES = NUMERIC + ['wine_type']
UNITS = {
    'fixed_acidity': 'g/L', 'volatile_acidity': 'g/L', 'citric_acid': 'g/L',
    'residual_sugar': 'g/L', 'chlorides': 'g/L', 'free_sulfur_dioxide': 'mg/L',
    'total_sulfur_dioxide': 'mg/L', 'density': 'g/cm³', 'pH': 'pH units',
    'sulphates': 'g/L', 'alcohol': '% by volume',
}

def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy, standardize labels, and coerce bad numeric cells to missing."""
    result = frame.copy(deep=True)
    result.columns = result.columns.str.strip().str.replace(' ', '_')
    missing = set(FEATURES) - set(result.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    for name in NUMERIC:
        result[name] = pd.to_numeric(result[name], errors='coerce').replace(
            [np.inf, -np.inf], np.nan)
    result['wine_type'] = result['wine_type'].map(
        lambda x: x.strip().lower() if isinstance(x, str) else np.nan)
    if not result['wine_type'].dropna().isin(['red', 'white']).all():
        raise ValueError('wine_type must be red or white')
    return result

def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), NUMERIC),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), ['wine_type']),
    ])

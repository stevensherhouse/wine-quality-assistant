import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.preprocess import NUMERIC, clean_frame, make_preprocessor

def sample():
    data = {c: [1., 2., 3., 4.] for c in NUMERIC}
    data['wine_type'] = ['red', 'white', 'red', 'white']
    return pd.DataFrame(data)

def test_missing_values_imputed():
    frame = sample()
    frame.loc[0, NUMERIC[0]] = np.nan
    frame.loc[1, 'wine_type'] = np.nan
    assert np.isfinite(make_preprocessor().fit_transform(frame)).all()

def test_categoricals_encoded():
    result = make_preprocessor().fit_transform(sample())
    assert result.shape == (4, 13)
    np.testing.assert_array_equal(result[:, -2:], [[1, 0], [0, 1], [1, 0], [0, 1]])

def test_numeric_scaling():
    result = make_preprocessor().fit_transform(sample())[:, :len(NUMERIC)]
    np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(result.std(axis=0), 1, atol=1e-10)

def test_original_frame_is_unchanged():
    frame = sample()
    frame.loc[0, 'fixed_acidity'] = np.inf
    original = frame.copy(deep=True)
    cleaned = clean_frame(frame)
    make_preprocessor().fit_transform(cleaned)
    assert_frame_equal(frame, original)
    assert np.isnan(cleaned.loc[0, 'fixed_acidity'])

def test_transform_does_not_refit_imputer():
    frame = sample()
    prep = make_preprocessor().fit(frame)
    inference = frame.iloc[[0]].copy()
    inference['fixed_acidity'] = np.nan
    np.testing.assert_allclose(prep.transform(inference)[0, 0], 0, atol=1e-10)

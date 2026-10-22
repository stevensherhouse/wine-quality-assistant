import numpy as np
import yaml
from src.evaluate import metrics
from src.preprocess import FEATURES
from tests.conftest import ROOT

def test_prediction_type_shape(trained):
    model, test, _ = trained
    predictions = model.predict(test[FEATURES].head(7))
    assert predictions.shape == (7,)
    assert np.issubdtype(predictions.dtype, np.floating)
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 10)).all()

def test_held_out_performance_threshold(trained):
    model, test, metadata = trained
    scores = metrics(test.quality, model.predict(test[FEATURES]))
    config = yaml.safe_load((ROOT / 'configs/config.yaml').read_text())
    assert scores['r2'] >= config['minimum_test_r2']
    assert scores['rmse'] < metadata['baseline_test_metrics']['rmse']
    assert abs(scores['mae'] - metadata['test_metrics']['mae']) < 1e-10

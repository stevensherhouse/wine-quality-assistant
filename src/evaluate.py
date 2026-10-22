"""Regression metrics and MLflow experiment comparison."""
import argparse
import json
import os
from pathlib import Path
import mlflow
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

def metrics(y_true, predictions) -> dict:
    mse = mean_squared_error(y_true, predictions)
    return {'mae': float(mean_absolute_error(y_true, predictions)),
            'mse': float(mse), 'rmse': float(np.sqrt(mse)),
            'r2': float(r2_score(y_true, predictions))}

def setup_tracking():
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI') or
                            'sqlite:///' + (ROOT / 'mlflow.db').as_posix())

def compare_runs(experiment_name, batch_id=None):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError('No experiment found; run training first')
    query = "attributes.status = 'FINISHED'"
    if batch_id:
        query += f" and tags.batch_id = '{batch_id}'"
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
        filter_string=query, order_by=['metrics.val_rmse ASC'])
    if runs.empty:
        raise ValueError('No finished experiment runs found')
    return runs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs/config.yaml'))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    setup_tracking()
    # The selected batch avoids mixing incomparable historical data/splits.
    selection = json.loads((ROOT / config['model_dir'] / 'metadata.json').read_text())
    runs = compare_runs(config['experiment_name'], selection['batch_id'])
    print(runs[['run_id', 'params.config_name', 'metrics.val_rmse',
                'metrics.test_rmse', 'metrics.test_mae', 'metrics.test_r2']].to_string(index=False))
    print('Best validation run:', runs.iloc[0]['run_id'])

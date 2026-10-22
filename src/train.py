"""Five tracked configurations; choose by validation RMSE, report held-out tests."""
import argparse
import hashlib
import json
import uuid
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from mlflow.models import infer_signature
from src.data import download_data, load_data
from src.evaluate import ROOT, compare_runs, metrics, setup_tracking
from src.preprocess import FEATURES, NUMERIC, make_preprocessor

def train(config_path: Path, download=False):
    config = yaml.safe_load(config_path.read_text())
    directory = ROOT / config['data_dir']
    if download:
        download_data(directory)
    data, version = load_data(directory)
    version_hash = hashlib.sha256(json.dumps(version, sort_keys=True).encode()).hexdigest()
    seed = config['seed']
    trainval, test = train_test_split(data, test_size=config['test_size'],
                                     random_state=seed, stratify=data.wine_type)
    trainset, validation = train_test_split(trainval,
        test_size=config['validation_size'] / (1 - config['test_size']),
        random_state=seed, stratify=trainval.wine_type)
    X, y = trainset[FEATURES], trainset.quality
    setup_tracking()
    mlflow.set_experiment(config['experiment_name'])
    batch = uuid.uuid4().hex
    report_dir = ROOT / config['report_dir']
    model_dir = ROOT / config['model_dir']
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    classes = {'ridge': Ridge, 'random_forest': RandomForestRegressor,
               'extra_trees': ExtraTreesRegressor}
    fitted = {}
    for spec in config['models']:
        params = dict(spec['params'])
        if spec['algorithm'] != 'ridge':
            params['random_state'] = seed
        pipeline = Pipeline([('preprocess', make_preprocessor()),
                             ('model', classes[spec['algorithm']](**params))])
        with mlflow.start_run(run_name=spec['name'], tags={'batch_id': batch}) as run:
            pipeline.fit(X, y)
            mlflow.log_params({
                'config_name': spec['name'], 'algorithm': spec['algorithm'],
                'data_version': version_hash, 'seed': seed,
                'test_size': config['test_size'], 'validation_size': config['validation_size'],
                'train_rows': len(X), 'val_rows': len(validation), 'test_rows': len(test),
            })
            # Includes defaults and preprocessing settings, not just overridden values.
            mlflow.log_params({k: str(v) for k, v in pipeline.get_params(deep=True).items()})
            mlflow.log_dict(version, 'data_version.json')
            mlflow.log_dict(config, 'training_config.json')
            for label, split in [('val', validation), ('test', test)]:
                scores = metrics(split.quality, pipeline.predict(split[FEATURES]))
                mlflow.log_metrics({f'{label}_{k}': v for k, v in scores.items()})
            mlflow.sklearn.log_model(pipeline, artifact_path='model',
                signature=infer_signature(X, pipeline.predict(X)), input_example=X.head(2),
                pip_requirements=['scikit-learn==1.6.1', 'pandas==2.2.3', 'numpy==2.2.5'])
            fitted[run.info.run_id] = pipeline
    runs = compare_runs(config['experiment_name'], batch)
    best = runs.iloc[0]
    pipeline = fitted[best.run_id]
    joblib.dump(pipeline, model_dir / 'best_model.joblib')
    test.to_csv(model_dir / 'test_sample.csv', index=False)
    metadata = {
        'batch_id': batch, 'run_id': best.run_id,
        'config_name': best['params.config_name'], 'data': version,
        'selection_rule': 'Lowest validation RMSE; test metrics never used for ranking',
        'split_rows': {'train': len(X), 'validation': len(validation), 'test': len(test)},
        'test_metrics': metrics(test.quality, pipeline.predict(test[FEATURES])),
        'bounds': {c: [float(X[c].min()), float(X[c].max())] for c in NUMERIC},
        'baseline_test_metrics': metrics(test.quality,
            DummyRegressor().fit(X, y).predict(test[FEATURES])),
    }
    (model_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (report_dir / 'results.json').write_text(json.dumps(metadata, indent=2))
    columns = ['run_id', 'params.config_name'] + [
        f'metrics.{split}_{metric}' for split in ['val', 'test']
        for metric in ['mae', 'mse', 'rmse', 'r2']]
    runs[columns].to_csv(report_dir / 'experiments.csv', index=False)
    print(json.dumps(metadata, indent=2))
    return metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/config.yaml')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    train(args.config, args.download)

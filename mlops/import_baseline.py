"""Register the EXISTING baseline in MLflow without retraining or invented measurements."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def import_existing_baseline(root: Path, tracking_uri: str | None = None) -> str:
    import mlflow
    import mlflow.sklearn
    import joblib

    artifacts = root / 'artifacts'
    metadata = json.loads((artifacts / 'baseline_metadata.json').read_text(encoding='utf-8'))
    pipeline = joblib.load(artifacts / 'baseline_model.joblib')
    if set(pipeline.classes_) != {'neutral', 'offensive', 'toxic'}:
        raise ValueError('У baseline изменена схема классов')
    mlflow.set_tracking_uri(tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns'))
    mlflow.set_experiment('team12-toxicity')
    with mlflow.start_run(run_name='import-existing-baseline') as run:
        mlflow.log_params({'algorithm': metadata['model'], 'seed': metadata['seed'],
                           'dataset_sha256': metadata['dataset_sha256'],
                           'train_rows': metadata['train_rows'],
                           'test_rows': metadata['test_rows']})
        mlflow.log_metrics({'test_macro_f1': metadata['test_macro_f1'],
                            'test_toxic_recall': metadata['test_toxic_recall']})
        mlflow.set_tags({'stage': 'existing_baseline', 'source': 'archived_metadata_not_recomputed'})
        for name in ['baseline_metadata.json', 'test_report.csv', 'validation_comparison.csv']:
            mlflow.log_artifact(str(artifacts / name), artifact_path='existing_reports')
        mlflow.sklearn.log_model(pipeline, name='model')
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking-uri', default=None)
    args = parser.parse_args()
    print('MLflow run:', import_existing_baseline(Path(__file__).resolve().parent.parent, args.tracking_uri))


if __name__ == '__main__':
    main()

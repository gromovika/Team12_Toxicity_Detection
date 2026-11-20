"""Checks that real MLflow runs and their metadata are available on the server."""
from __future__ import annotations

import argparse
import os


def verify(require_candidate: bool = False, tracking_uri: str | None = None) -> dict:
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name('team12-toxicity')
    if exp is None:
        raise RuntimeError('MLflow experiment team12-toxicity отсутствует')
    runs = client.search_runs([exp.experiment_id], max_results=500)
    baselines = [r for r in runs if r.data.tags.get('stage') == 'existing_baseline']
    if not baselines:
        raise RuntimeError('В MLflow нет импортированного baseline')
    baseline = baselines[0]
    for key in ('test_macro_f1', 'test_toxic_recall'):
        if key not in baseline.data.metrics:
            raise RuntimeError(f'Baseline в MLflow без метрики {key}')
    if not client.list_artifacts(baseline.info.run_id, path='existing_reports'):
        raise RuntimeError('В MLflow не обнаружены исходные baseline-отчёты')
    champions = [r for r in runs if r.data.tags.get('stage') == 'candidate_not_promoted']
    if require_candidate and not champions:
        raise RuntimeError('В MLflow нет обученного кандидата, запустите train-fast')
    if require_candidate:
        for key in ('test_macro_f1','test_toxic_recall','test_accuracy'):
            if key not in champions[0].data.metrics:
                raise RuntimeError(f'Кандидат в MLflow без метрики {key}')
    return {
        'tracking_uri': uri,
        'experiment_id': exp.experiment_id,
        'baseline_run_id': baseline.info.run_id,
        'baseline_macro_f1': baseline.data.metrics['test_macro_f1'],
        'candidate_run_id': champions[0].info.run_id if champions else None,
        'candidate_macro_f1': champions[0].data.metrics['test_macro_f1'] if champions else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking-uri')
    parser.add_argument('--require-candidate', action='store_true')
    args = parser.parse_args()
    print(verify(args.require_candidate, args.tracking_uri))


if __name__ == '__main__':
    main()

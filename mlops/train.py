"""Train candidates against the original split, log experiments to MLflow.

Never overwrites the existing production baseline. Requires the source dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from mlops.data import CLASSES, dataset_sha256, read_labeled_dataset, split_like_original


def make_pipeline(name: str, seed: int = 42):
    models = {
        'Naive Bayes': lambda: MultinomialNB(alpha=1.0),
        'Logistic Regression': lambda: LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=seed),
        'Linear SVM (SGD)': lambda: CalibratedClassifierCV(
            SGDClassifier(loss='hinge', class_weight='balanced', random_state=seed,
                          max_iter=1000, tol=1e-3), method='sigmoid', cv=3),
    }
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=.98,
                                  max_features=100000, sublinear_tf=True)),
        ('clf', models[name]()),
    ])


def write_artifact(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.joblib', delete=False) as handle:
            tmp = Path(handle.name)
        joblib.dump(model, tmp)
        os.replace(tmp, path)
    finally:
        if tmp is not None: tmp.unlink(missing_ok=True)


def train_candidates(data_path: str | Path, output: str | Path = 'artifacts/candidate_model.joblib',
                     seed: int = 42, tracking_uri: str | None = None,
                     log_mlflow: bool = True, models: tuple[str, ...] | None = None) -> dict:
    data_path, output = Path(data_path), Path(output)
    allowed = ('Naive Bayes', 'Logistic Regression', 'Linear SVM (SGD)')
    selected = tuple(models) if models is not None else allowed
    if not selected or any(name not in allowed for name in selected) or len(set(selected)) != len(selected):
        raise ValueError(f'--models должен содержать неповторяющиеся имена из: {allowed}')
    data, malformed = read_labeled_dataset(data_path)
    train, val, test = split_like_original(data, seed)
    mlflow = None
    if log_mlflow:
        import mlflow as tracking
        import mlflow.sklearn
        mlflow = tracking
        mlflow.set_tracking_uri(tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns'))
        mlflow.set_experiment('team12-toxicity')
    fitted = {}
    comparison = []
    sha = dataset_sha256(data_path)
    for name in selected:
        model = make_pipeline(name, seed)
        model.fit(train.text_clean, train.label)
        pred = model.predict(val.text_clean)
        scores = {
            'accuracy_val': float(accuracy_score(val.label, pred)),
            'macro_f1_val': float(f1_score(val.label, pred, average='macro', labels=CLASSES, zero_division=0)),
            'weighted_f1_val': float(f1_score(val.label, pred, average='weighted', labels=CLASSES, zero_division=0)),
        }
        fitted[name] = model
        comparison.append({'model': name, **scores})
        if mlflow is not None:
            with mlflow.start_run(run_name=f'candidate-{name}'):
                mlflow.log_params({'algorithm': name, 'seed': seed, 'dataset_sha256': sha,
                                   'train_rows': len(train), 'validation_rows': len(val),
                                   'ngram_range': '(1,2)', 'max_features': 100000,
                                   'probability_calibration': 'sigmoid_cv3' if name == 'Linear SVM (SGD)' else 'none'})
                mlflow.log_metrics(scores)
                mlflow.set_tags({'stage': 'validation', 'split': 'original_70_15_15'})
    comparison.sort(key=lambda row: row['macro_f1_val'], reverse=True)
    best_name = comparison[0]['model']
    champion = fitted[best_name]
    if not hasattr(champion, 'predict_proba'):
        raise RuntimeError('Кандидат несовместим с API: нет predict_proba')
    # The holdout is used ONCE, only after the champion is picked on validation.
    predictions = champion.predict(test.text_clean)
    report = classification_report(test.label, predictions, labels=CLASSES,
                                   output_dict=True, zero_division=0)
    metadata = {
        'model': best_name, 'classes': list(CLASSES), 'seed': seed,
        'train_rows': len(train), 'val_rows': len(val), 'test_rows': len(test),
        'test_macro_f1': float(report['macro avg']['f1-score']),
        'test_toxic_recall': float(report['toxic']['recall']),
        'dataset_sha256': sha, 'malformed_rows': malformed,
        'candidate': True, 'validation_comparison': comparison,
        'models_compared': list(selected),
    }
    # Candidate is deliberately a plain sklearn Pipeline, exactly as app/model_service.py expects.
    write_artifact(champion, output)
    metadata_path = output.with_name(output.stem.replace('_model','') + '_metadata.json')
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    if mlflow is not None:
        with mlflow.start_run(run_name=f'champion-{best_name}'):
            mlflow.log_params({'algorithm': best_name, 'seed': seed, 'dataset_sha256': sha,
                               'train_rows': len(train), 'test_rows': len(test)})
            mlflow.log_metrics({'test_macro_f1': metadata['test_macro_f1'],
                                'test_toxic_recall': metadata['test_toxic_recall'],
                                'test_accuracy': float(accuracy_score(test.label, predictions))})
            mlflow.set_tags({'stage': 'candidate_not_promoted', 'split': 'original_70_15_15'})
            mlflow.log_artifact(str(metadata_path), artifact_path='reports')
            mlflow.sklearn.log_model(champion, name='model')
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to original dataset.txt or ZIP')
    parser.add_argument('--output', default='artifacts/candidate_model.joblib')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tracking-uri', default=None)
    parser.add_argument('--no-mlflow', action='store_true', help='Offline smoke test')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Optional quoted algorithm names. Example: --models \"Naive Bayes\"')
    args = parser.parse_args()
    result = train_candidates(args.data, args.output, args.seed, args.tracking_uri,
                              not args.no_mlflow, models=tuple(args.models) if args.models else None)
    print(json.dumps({'model': result['model'], 'test_macro_f1': result['test_macro_f1']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Re-evaluate the trusted archived baseline on its original holdout, with no data leakage.

Outputs only aggregated evaluation statistics (never original comments).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from mlops.data import CLASSES, dataset_sha256, read_labeled_dataset, split_like_original


def evaluate(data_path: str | Path, root: str | Path = '.') -> dict:
    root = Path(root)
    metadata = json.loads((root / 'artifacts/baseline_metadata.json').read_text(encoding='utf-8'))
    file_hash = dataset_sha256(data_path)
    if file_hash != metadata['dataset_sha256']:
        raise ValueError('SHA256 датасета не совпадает с baseline metadata')
    frame, excluded = read_labeled_dataset(data_path)
    train, validation, test = split_like_original(frame, seed=int(metadata['seed']))
    for name, data, key in (
        ('train', train, 'train_rows'),
        ('validation', validation, 'val_rows'),
        ('test', test, 'test_rows'),
    ):
        if len(data) != metadata[key]:
            raise ValueError(f'Размер {name} не совпадает с исходным: {len(data)} != {metadata[key]}')
    # This joblib file is trusted from the user's own project; untrusted joblib files are unsafe.
    model = joblib.load(root / 'artifacts/baseline_model.joblib')
    if set(model.classes_) != set(CLASSES):
        raise ValueError('Несовместимый набор классов модели')
    truth = test.label.to_numpy()
    predicted = model.predict(test.text_clean)
    report = classification_report(truth, predicted, labels=CLASSES, output_dict=True, zero_division=0)
    scores = {
        'accuracy': float(accuracy_score(truth, predicted)),
        'macro_f1': float(f1_score(truth, predicted, labels=CLASSES, average='macro', zero_division=0)),
        'toxic_recall': float(report['toxic']['recall']),
    }
    model_metrics_match = {
        'macro_f1': math.isclose(scores['macro_f1'], metadata['test_macro_f1'], abs_tol=1e-12, rel_tol=0),
        'toxic_recall': math.isclose(scores['toxic_recall'], metadata['test_toxic_recall'], abs_tol=1e-12, rel_tol=0),
    }
    return {
        'dataset_sha256': file_hash,
        'model': metadata['model'],
        'split': {'train': len(train), 'validation': len(validation), 'test': len(test)},
        'original_parser_excluded_rows': excluded,
        'labels': list(CLASSES),
        'scores': scores,
        'per_class': {label: {key: float(value) for key, value in report[label].items()}
                      for label in CLASSES},
        'confusion_matrix': confusion_matrix(truth, predicted, labels=CLASSES).tolist(),
        'matches_archived_report': model_metrics_match,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/dataset.txt.zip')
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', default='reports/baseline_real_dataset_evaluation.json')
    args = parser.parse_args()
    result = evaluate(args.data, args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(result['matches_archived_report'].values()):
        raise SystemExit('Проверка метрик baseline не пройдена')


if __name__ == '__main__':
    main()

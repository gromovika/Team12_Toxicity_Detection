import json
from pathlib import Path

import joblib
import pytest

from mlops.data import CLASSES, dataset_sha256, read_labeled_dataset, split_like_original
from mlops.train import train_candidates


def test_parse_and_leakage(tmp_path):
    rows=[]
    for idx in range(35):
        rows += [f'__label__NORMAL спасибо помощь {idx} нормально',
                 f'__label__INSULT обидно неприятно {idx} глупый',
                 f'__label__THREAT угроза опасность {idx} угрожаю']
    rows += ['__label__NORMAL повторяющийся текст', '__label__THREAT повторяющийся текст',
             'неверная строка', '__label__NORMAL спасибо помощь 0 нормально']
    path=tmp_path/'dataset.txt'
    path.write_text('\n'.join(rows)+'\n',encoding='utf-8')
    frame,bad=read_labeled_dataset(path)
    assert bad==1
    train,val,test=split_like_original(frame)
    assert not (set(train.text_clean)&set(test.text_clean))
    assert not (set(val.text_clean)&set(test.text_clean))
    assert 'повторяющийся текст' not in set(train.text_clean)|set(val.text_clean)|set(test.text_clean)
    assert set(train.label)==set(CLASSES)
    assert len(dataset_sha256(path))==64


def test_training_candidate_matches_original_api(tmp_path):
    path=tmp_path/'dataset.txt'
    path.write_text(''.join(
        f'__label__{lab} {word} тестовые примеры номер {i} уникальность\n'
        for i in range(45)
        for lab,word in [('NORMAL','добро отлично'),('INSULT','ругательство грубо'),('THREAT','страшная угроза')]
    ),encoding='utf8')
    out=tmp_path/'candidate_model.joblib'
    metadata=train_candidates(path,out,log_mlflow=False)
    assert metadata['model'] in ('Logistic Regression','Naive Bayes')
    pipeline=joblib.load(out)
    assert set(pipeline.classes_)==set(CLASSES)
    assert len(pipeline.predict_proba(['Спасибо за отзыв'])[0])==3
    assert metadata['test_rows']>0
    assert out.with_name('candidate_metadata.json').is_file()
    assert metadata['dataset_sha256']==dataset_sha256(path)
    assert not (tmp_path/'baseline_model.joblib').exists()


def test_zip_hash_and_multilabel_semantics(tmp_path):
    import hashlib
    import zipfile
    text = ("__label__NORMAL корректный текст\n"
            "__label__INSULT,__label__THREAT одновременно две метки\n"
            "__label__THREAT угроза\n")
    raw = tmp_path / 'dataset.txt'
    raw.write_bytes(text.encode('utf-8'))
    zipped = tmp_path / 'dataset.txt.zip'
    with zipfile.ZipFile(zipped, 'w') as archive:
        archive.write(raw, arcname='dataset.txt')
    assert dataset_sha256(zipped) == dataset_sha256(raw) == hashlib.sha256(raw.read_bytes()).hexdigest()
    data, skipped = read_labeled_dataset(zipped)
    assert len(data) == 2
    assert skipped == 1  # Reproduce original notebook, which omits multi-label entries.


def test_zip_requires_one_dataset_member(tmp_path):
    import zipfile
    archive_path = tmp_path / 'two.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('part1.txt', '__label__NORMAL один\n')
        archive.writestr('part2.txt', '__label__NORMAL два\n')
    with pytest.raises(ValueError, match='ровно один'):
        dataset_sha256(archive_path)


def test_calibrated_svm_serving_compatible(tmp_path):
    from mlops.train import make_pipeline
    from sklearn.datasets import make_classification
    # A calibrated linear SVM must export probabilities, not only margins.
    sentences = [f'слово {label} пример {i}' for label in ('neutral','offensive','toxic') for i in range(20)]
    labels = [label for label in ('neutral','offensive','toxic') for _ in range(20)]
    model = make_pipeline('Linear SVM (SGD)')
    model.fit(sentences, labels)
    proba = model.predict_proba(['слово neutral пример 9'])[0]
    assert len(proba) == 3
    assert abs(sum(proba) - 1.0) < 1e-6

def test_choose_one_fast_model(tmp_path):
    path = tmp_path/'data.txt'
    path.write_text(''.join(
        f'__label__{lab} {word} пример уникальный {i}\n'
        for i in range(40)
        for lab,word in [('NORMAL','спасибо'),('INSULT','грубиян'),('THREAT','угроза')]
    ), encoding='utf-8')
    out = tmp_path/'candidate_model.joblib'
    metadata = train_candidates(path, out, models=('Naive Bayes',), log_mlflow=False)
    assert metadata['models_compared'] == ['Naive Bayes']
    assert metadata['model'] == 'Naive Bayes'
    assert out.is_file()

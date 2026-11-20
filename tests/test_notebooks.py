import json
import os
from pathlib import Path

import joblib
import nbformat
import pytest
from nbclient import NotebookClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlops.data import dataset_sha256, read_labeled_dataset, split_like_original

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name',['01_quality_evaluation.ipynb','02_robustness_testing.ipynb'])
def test_notebook_runs_on_synthetic_data(tmp_path,monkeypatch,name):
    data=tmp_path/'dataset.txt'
    data.write_text(''.join(
        f'__label__{lab} {word} пример {i} текст\n'
        for i in range(35)
        for lab,word in [('NORMAL','благодарю отлично'),('INSULT','грубиян обидно'),('THREAT','угрожаю опасность')]
    ),encoding='utf8')
    frame,_=read_labeled_dataset(data)
    train,val,test=split_like_original(frame)
    model=Pipeline([('tfidf',TfidfVectorizer()),('clf',LogisticRegression(max_iter=200))])
    model.fit(train.text_clean,train.label)
    model_path=tmp_path/'baseline_model.joblib'
    joblib.dump(model,model_path)
    from sklearn.metrics import f1_score, recall_score
    pred=model.predict(test.text_clean)
    metadata={'dataset_sha256':dataset_sha256(data),'model':'Logistic Regression','seed':42,
              'train_rows':len(train),'val_rows':len(val),'test_rows':len(test),
              'test_macro_f1':float(f1_score(test.label,pred,average='macro')),
              'test_toxic_recall':float(recall_score(test.label,pred,labels=['toxic'],average='macro'))}
    meta_path=tmp_path/'baseline_metadata.json'
    meta_path.write_text(json.dumps(metadata),encoding='utf8')
    monkeypatch.setenv('DATA_PATH',str(data))
    monkeypatch.setenv('MODEL_PATH',str(model_path))
    monkeypatch.setenv('MODEL_METADATA_PATH',str(meta_path))
    nb=nbformat.read(ROOT/'notebooks'/name,as_version=4)
    NotebookClient(nb,timeout=90,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()

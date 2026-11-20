"""Actual original holdout check when user's private dataset is locally available."""
from pathlib import Path
import pytest
from mlops.evaluate_baseline import evaluate

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/dataset.txt.zip'

@pytest.mark.skipif(not DATA.exists(), reason='User-supplied dataset not published in repository')
def test_original_baseline_real_holdout():
    report=evaluate(DATA, ROOT)
    assert report['dataset_sha256']=='426e6bd2514fb974b68c3794da6862f5c67a34f32bdd41efeb5266a101048381'
    assert report['split']=={'train':167930,'validation':35985,'test':35985}
    assert report['original_parser_excluded_rows']==8333
    assert all(report['matches_archived_report'].values())

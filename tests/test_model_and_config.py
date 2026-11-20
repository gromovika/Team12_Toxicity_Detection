import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'app'))
from model_service import ModelService


def test_baseline_artifact_loads_and_predicts():
    service=ModelService(REPO/'artifacts/baseline_model.joblib',REPO/'artifacts/baseline_metadata.json')
    service.load()
    assert service.loaded
    assert service.classes==['neutral','offensive','toxic']
    result=service.predict('Добрый день, как ваши дела?')
    assert result['label'] in service.classes
    assert abs(sum(result['probabilities'].values())-1.0)<1e-6
    assert service.metadata['test_macro_f1']>0.8  # stored metadata, not recalculated

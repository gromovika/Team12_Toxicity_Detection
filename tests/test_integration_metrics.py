"""HTTP smoke without lifespan: substitutes aiogram only if absent, DB is fake.

Integration with real PostgreSQL/Telegram is covered by app/tests/test_api.py,
not by this isolated route-level test.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'app'))


def stub_aiogram():
    if importlib.util.find_spec('aiogram') is not None:
        return
    class Filter:
        def __getattr__(self, _):return self
        def __or__(self, _):return self
    class Router:
        def message(self,*args,**kwargs): return lambda func:func
    class Dispatcher:
        def include_router(self,*args):pass
    class Bot:pass
    aiogram=types.ModuleType('aiogram')
    aiogram.Bot=Bot;aiogram.Dispatcher=Dispatcher;aiogram.Router=Router;aiogram.F=Filter()
    filters=types.ModuleType('aiogram.filters')
    filters.Command=lambda *_:Filter()
    filters.CommandStart=lambda *_:Filter()
    atypes=types.ModuleType('aiogram.types')
    atypes.Message=type('Message',(),{})
    sys.modules.update({'aiogram':aiogram,'aiogram.filters':filters,'aiogram.types':atypes})


def test_real_api_routes_predict_and_monitoring(monkeypatch):
    stub_aiogram()
    import main
    main.service.load()
    class FakeDB:
        def ping(self):return True
        def save_prediction(self,**kwargs):pass
    monkeypatch.setattr(main,'db',FakeDB())
    monkeypatch.setattr(main,'model_version_id',1)
    # Do not enter TestClient context, which starts PostgreSQL lifespan.
    client=TestClient(main.app)
    assert client.get('/health/live').status_code==200
    assert client.get('/health/ready').status_code==200
    prediction=client.post('/predict',json={'text':'Привет, как дела?'})
    assert prediction.status_code==200
    assert set(prediction.json()['probabilities'])=={'neutral','offensive','toxic'}
    assert client.post('/predict',json={'text':'    '}).status_code==422
    metrics=client.get('/metrics')
    assert metrics.status_code==200
    assert 'toxicity_http_requests_total' in metrics.text
    assert 'toxicity_predictions_total' in metrics.text
    assert 'toxicity_inference_duration_seconds' in metrics.text
    monkeypatch.setattr(main,'db',None)
    assert client.get('/health/ready').status_code==503

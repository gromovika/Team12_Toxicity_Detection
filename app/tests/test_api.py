import pytest

import main
from main import build_reply, validate_text, TextVerdict


class FakeMsg:
    def __init__(self, text=None):
        self.text = text


def test_healthcheck(client):
    r = client.get("/healthcheck")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["classes"] == ["neutral", "offensive", "toxic"]


def test_predict_endpoint(client):
    r = client.post("/predict", json={"text": "ты идиот, заткнись"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "offensive"
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


@pytest.mark.parametrize(
    "text, expected",
    [
        ("привет, как дела?", TextVerdict.OK),
        ("x", TextVerdict.TOO_SHORT),
        ("!!!@@@###", TextVerdict.GIBBERISH),
        ("😀😀😀😀", TextVerdict.GIBBERISH),
        (None, TextVerdict.NOT_TEXT),
    ],
)
def test_validate_text(text, expected):
    assert validate_text(FakeMsg(text))[0] == expected


def test_build_reply_direct_call():
    reply = build_reply("ты идиот, заткнись")
    assert "оскорбительный" in reply
    assert "%" in reply


def test_predict_toxic(client):
    r = client.post("/predict", json={"text": "тебя надо убить, ублюдок"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "toxic"
    assert body["probabilities"]["toxic"] > 0.9


def test_predict_too_long(client):
    r = client.post("/predict", json={"text": "а" * 5000})
    assert r.status_code == 422


@pytest.mark.parametrize(
    "text",
    [
        "првиет, ка кдела",          # опечатки
        "ААААА это просто невыносимо",  # повторяющиеся символы
        "ЁЖИКИ В ТУМАНЕ",            # регистр + ё
        "нормальн такой комментарий",   # разговорное написание
    ],
)
def test_predict_robustness(client, text):
    r = client.post("/predict", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"neutral", "offensive", "toxic"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_predict_latency(client):
    import time

    start = time.perf_counter()
    r = client.post("/predict", json={"text": "проверка скорости обработки сообщения"})
    elapsed = time.perf_counter() - start
    assert r.status_code == 200
    assert elapsed < 1.0, f"predict занял {elapsed:.2f}s"


def test_db_model_version_registered(client):
    versions = main.db.model_versions()
    assert len(versions) == 1
    assert versions[0].version == main.settings.model_version
    assert versions[0].model_name == "Logistic Regression"


def test_db_prediction_saved(client):
    before = len(main.db.recent_predictions(limit=100))
    r = client.post("/predict", json={"text": "привет, как дела?"})
    assert r.status_code == 200
    rows = main.db.recent_predictions(limit=100)
    assert len(rows) == before + 1
    row = rows[0]  # последняя запись
    assert row.text == "привет, как дела?"
    assert row.predicted_class == "neutral"
    assert row.source == "api"
    assert row.true_class is None  # до ручной проверки пусто
    assert row.model_version_id == main.model_version_id
    assert 0.0 <= row.confidence <= 1.0


def test_db_bot_prediction_source(client):
    before = len(main.db.recent_predictions(limit=100))
    build_reply("ты идиот, заткнись")
    rows = main.db.recent_predictions(limit=100)
    assert len(rows) == before + 1
    assert rows[0].source == "telegram"

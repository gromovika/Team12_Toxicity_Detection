import os

import pytest
from fastapi.testclient import TestClient

import main
from db import PSYCOPG2_URL_PREFIX, Database
from main import app, build_reply, validate_text, TextVerdict, service


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Uses a dedicated real PostgreSQL when TEAM12_TEST_DB_URL is set.

    Otherwise retains the original embedded PostgreSQL fixture.
    Never point TEAM12_TEST_DB_URL to a real/staging/production database.
    """
    import dataclasses

    original_settings = main.settings
    test_db_url = os.getenv('TEAM12_TEST_DB_URL', '').strip()
    server = None
    try:
        if test_db_url:
            from sqlalchemy.engine import make_url
            parsed_url = make_url(test_db_url)
            if parsed_url.host not in ('127.0.0.1', 'localhost') or parsed_url.database != 'toxicity_test':
                raise ValueError('TEAM12_TEST_DB_URL должен указывать только на локальную БД toxicity_test')
            main.settings = dataclasses.replace(
                main.settings, bot_token='', db_url=test_db_url,
                model_version='baseline',
            )
        else:
            import pgserver
            # Original embedded-PostgreSQL behavior (no Docker required).
            for attempt in range(3):
                try:
                    server = pgserver.get_server(str(tmp_path_factory.mktemp('pg')))
                    break
                except Exception:
                    if attempt == 2:
                        raise
            main.settings = dataclasses.replace(
                main.settings, bot_token='',
                db_url=server.get_uri().replace('postgresql://', PSYCOPG2_URL_PREFIX, 1),
                model_version='baseline',
            )
        with TestClient(app) as c:
            yield c
    finally:
        main.settings = original_settings
        if server is not None:
            server.cleanup()


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

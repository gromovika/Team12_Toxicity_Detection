import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

import main
from db import Database
from main import app, build_reply, validate_text, TextVerdict, service

# Отдельная тестовая база; по умолчанию — compose-база на localhost:5432.
# Для другого сервера задайте TEST_DB_URL.
TEST_DB_URL = os.environ.get(
    "TEST_DB_URL",
    "postgresql://postgres:postgres@localhost:55432/toxicity_test",
)


def _ensure_database(url: str) -> None:
    """Создаёт тестовую базу, если её нет (нужны права на CREATE DATABASE)."""
    u = make_url(url).set(drivername="postgresql+psycopg2")
    admin = create_engine(u.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": u.database}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{u.database}"'))
    admin.dispose()


@pytest.fixture(scope="module")
def client():
    # Бот в тестах не стартуем; тестовая база отдельная — lifespan подхватит
    # подменённый db_url и не тронет рабочую базу
    import dataclasses

    main.settings = dataclasses.replace(main.settings, bot_token="", db_url=TEST_DB_URL)
    _ensure_database(TEST_DB_URL)
    main.db = Database(TEST_DB_URL)
    main.db.init()
    main.db.reset()
    with TestClient(app) as c:
        yield c
    main.db.close()


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

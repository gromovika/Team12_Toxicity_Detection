import pytest
from fastapi.testclient import TestClient

import main
from main import app, build_reply, validate_text, TextVerdict, service


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


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


# ML-01: явная проверка класса toxic (самый редкий и трудный класс)
def test_predict_toxic(client):
    r = client.post("/predict", json={"text": "тебя надо убить, ублюдок"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "toxic"
    assert body["probabilities"]["toxic"] > 0.9


# BE-03: превышение допустимой длины текста
def test_predict_too_long(client):
    r = client.post("/predict", json={"text": "а" * 5000})
    assert r.status_code == 422


# ML-03: устойчивость к опечаткам, повторам, регистру и ё
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


# NFR-01: одиночный predict укладывается в разумное время
def test_predict_latency(client):
    import time

    start = time.perf_counter()
    r = client.post("/predict", json={"text": "проверка скорости обработки сообщения"})
    elapsed = time.perf_counter() - start
    assert r.status_code == 200
    assert elapsed < 1.0, f"predict занял {elapsed:.2f}s"

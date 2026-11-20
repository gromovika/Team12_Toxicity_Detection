# Mock-тесты: проверяем код вокруг модели, подменяя зависимости.

from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from main import app

FIXED_RESULT = {
    "label": "toxic",
    "probabilities": {"neutral": 0.1, "offensive": 0.2, "toxic": 0.7},
}


# Контракт ответа: с управляемым ответом «модели» проверяем структуру,
# не завися от того, что реально предскажет модель
def test_predict_contract_with_mocked_model(client):
    with patch.object(main.service, "predict", return_value=FIXED_RESULT):
        r = client.post("/predict", json={"text": "произвольный текст"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "toxic"
    assert body["probabilities"] == FIXED_RESULT["probabilities"]
    assert body["model"] == "Logistic Regression"


# Сбой «модели»: настоящую модель так не проверить — только подменой
def test_predict_model_failure(client):
    with patch.object(main.service, "predict", side_effect=RuntimeError("model down")):
        c = TestClient(app, raise_server_exceptions=False)
        with c:  # lifespan повторно — подхватит тестовую БД
            r = c.post("/predict", json={"text": "текст"})
    assert r.status_code == 500


# Сбой записи в БД не должен ронять ответ пользователю
def test_predict_db_failure_still_responds(client):
    with patch.object(main.db, "save_prediction", side_effect=Exception("db down")):
        r = client.post("/predict", json={"text": "привет, как дела?"})
    assert r.status_code == 200
    assert r.json()["label"] in {"neutral", "offensive", "toxic"}


# Форматирование ответа бота при известном результате модели
def test_build_reply_with_mocked_model(client):
    from main import build_reply

    with patch.object(main.service, "predict", return_value=FIXED_RESULT):
        reply = build_reply("что угодно")
    assert "токсичный" in reply
    assert "70.0%" in reply

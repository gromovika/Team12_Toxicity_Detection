# Метрики и логирование

def test_metrics_accumulate(client):
    before = client.get("/metrics").json()

    for _ in range(3):
        client.post("/predict", json={"text": "привет, как дела?"})
    client.post("/predict", json={"text": ""})  # 422 — должен попасть в ошибки

    after = client.get("/metrics").json()
    assert after["total_requests"] == before["total_requests"] + 5  # 4 запроса + этот /metrics
    assert after["total_errors"] == before["total_errors"] + 1
    assert after["avg_latency_ms"] > 0


def test_metrics_zero_division_safe(client):
    # error_rate и avg_latency не падают на нуле запросов в начале прогона
    body = client.get("/metrics").json()
    assert 0.0 <= body["error_rate"] <= 1.0
    assert body["avg_latency_ms"] >= 0.0

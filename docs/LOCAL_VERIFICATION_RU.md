# Протокол локальных проверок (Mac / VS Code)

Дата вывода: **20.11.2026**. Содержит только результаты, сообщённые командой из её терминала и скриншотов; **это не запись нового прогона на машине преподавателя**.

## Docker и PostgreSQL

1. Первый `bash scripts/local_validate.sh infra`: образы API/MLflow собраны, основной `db` — `Healthy`; API не стартовал из-за `Bind for 127.0.0.1:8000 failed: port is already allocated`.
2. Команда изменила внешний порт на `18000` и повторила `infra`: запуск **6 сервисов** (`db`, `db_test`, `api`, `mlflow`, `prometheus`, `grafana`). Первоначальная проверка readiness старым скриптом обращалась к занятому `8000` и давала `404`; внешняя привязка исправлена. В новой репозиторной версии вместо ручной замены используется `API_HOST_PORT`.
3. `bash scripts/local_validate.sh tests`:

```text
13 passed, 1 warning in 34.44s
19 passed, 1 warning in 2.71s
All tests passed; app/tests used actual isolated Docker PostgreSQL.
```

Предупреждения: `StarletteDeprecationWarning` в `fastapi.testclient`, не тестовые ошибки.

## MLflow

```text
Experiment: team12-toxicity (ID 1)
Baseline run: d253c58a84964bf2aea21cd48f8c1cc1
Naive Bayes run: 6794e67980ae4707967fc177b6e23319
Logistic Regression run: 565576dd2d16441fbca3925e8a0f36c9
Linear SVM (SGD) run: 995eed4129d94b04a1ec797ec177aa30
Champion Logistic Regression run: dbb8263e3fc4494491452039328ec311
```

Последний `compare`:

```text
test_macro_f1: baseline=0.857169; candidate=0.857169
test_toxic_recall: baseline=0.880342; candidate=0.880342
Candidate passes two metric checks. Manual review is still necessary.
```

Вывод: experiment logging и retraining работают. **Измеренного улучшения метрик нет**; модель не продвинута автоматически.

## Prometheus и Grafana

- Prometheus Targets: `toxicity-api`, endpoint `http://api:8000/metrics`, **1/1 up**, `UP`. [Скриншот от команды](evidence/prometheus_target_up.png).
- Grafana: `API availability = 1`. После тестовых POST-запросов появились линии `Prediction requests per second`, `Prediction latency p95` (около 9,6 мс в локальном снимке) и `Predicted class distribution` (`neutral`). [Скриншот от команды](evidence/grafana_predictions.png).
- Текущий факт не доказывает, что Alertmanager уведомляет об авариях. Это отдельный planned task.

## Как обновить доказательства для защиты

Перед загрузкой в GitHub выполните заново:

```bash
bash scripts/local_validate.sh infra
bash scripts/local_validate.sh status
bash scripts/local_validate.sh tests
bash scripts/local_validate.sh mlflow
bash scripts/local_validate.sh train-all
bash scripts/local_validate.sh compare
```

Приложите к отчёту/презентации (не содержащие секретов) снимки **GitHub Actions**, `pytest passed`, MLflow experiment runs, Prometheus Targets (UP), Grafana dashboard. Не публикуйте `.env`, токен Telegram, сырые комментарии и локальные БД.

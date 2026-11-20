# Изменения относительно присланного ZIP

## Сохранены без замены

- `artifacts/baseline_model.joblib` и все исходные отчёты.
- Исходные классы и формат ответа `/predict`.
- Telegram-бот, ORM-модели PostgreSQL, локальный режим с `pgserver`, старые проверки в `app/tests/test_api.py`.
- Ноутбуки первичного анализа и архитектуры БД; исходный экспериментальный ноутбук сохранён с одной опциональной секцией MLflow в конце.

## Изменённые исходные файлы

- `app/main.py` — безопасные метрики, новые health endpoints и согласованная с Telegram валидация входных текстов API.
- `app/db.py` — проверка доступности БД для `/health/ready`.
- `app/requirements.txt` — закреплена `scikit-learn==1.9.1` и добавлен `prometheus-client`.
- `Experiments_Data_Retraining 2.ipynb` — две ячейки опциональной отправки уже вычисленных результатов в MLflow, при `ENABLE_MLFLOW=1`.
- `README.md`, `app/README.md` — ссылки на новое MLOps-описание.

## Новые файлы

- `mlops/` — загрузка и сплит данных, обучение кандидатов с MLflow, импорт baseline в MLflow.
- `notebooks/` — отдельная оценка test c проверкой SHA-256 и диагностика устойчивости.
- `app/metrics.py` — Prometheus метрики без сырых пользовательских текстов.
- `Dockerfile`, `compose.yaml`, `.dockerignore`, `.env.example` (в корне) и requirements для deploy/MLops.
- `monitoring/` — Prometheus, alerts, Grafana provisioning и dashboard.
- `tests/` — независимые тесты Python и исполнение обоих новых ноутбуков.
- `.github/workflows/ci.yml` — CI для тестов и проверки Compose.
- `docs/AUDIT_RU.md`, `docs/DEPLOYMENT_STRATEGY_RU.md`, `README_MLOPS_RU.md`.

## Запуск и ограничения

Полный порядок действий приведён в [README_MLOPS_RU.md](README_MLOPS_RU.md). Не называйте компиляцию Python или парсинг YAML полноценным тестом Docker/MLflow: серверная проверка до выдачи архива не проводилась. Самого исходного обучающего датасета в ZIP нет.

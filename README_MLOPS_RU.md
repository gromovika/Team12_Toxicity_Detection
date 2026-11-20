> **Актуальная карта проекта и критериев:** [`README.md`](README.md), [`docs/SUBMISSION_CHECKLIST_RU.md`](docs/SUBMISSION_CHECKLIST_RU.md). Локальный Docker/PostgreSQL/MLflow подтверждён командой, включая 13+19 тестов: [`docs/LOCAL_VERIFICATION_RU.md`](docs/LOCAL_VERIFICATION_RU.md). Старые ограничения проверки в этой среде ниже сохранены для прозрачности истории.

> **Локальная проверка всех пунктов (Docker + реальный PostgreSQL + MLflow + обучение):** [`README_LOCAL_CHECK_RU.md`](README_LOCAL_CHECK_RU.md). Проверенные результаты переобучения: [`docs/RETRAINING_RESULTS_RU.md`](docs/RETRAINING_RESULTS_RU.md).

# Team12 — интегрированное MLOps-дополнение

**Важно:** это версия именно вашего ZIP-проекта, а не сторонний отдельный API. Сохранены оригинальные файлы модели, FastAPI, Telegram-бот, схема БД и учебные ноутбуки. Новые компоненты интегрированы в исходную архитектуру.

## Что реализовано

1. **MLflow:** `mlops/import_baseline.py`, `mlops/train.py`; опциональная ячейка в исходном ноутбуке. Run каждого алгоритма содержит параметры и validation, champion — тестовую метрику. Продукционная модель не меняется автоматически.
2. **Два ноутбука:** `notebooks/01_quality_evaluation.ipynb` (real baseline test только с исходным датасетом и проверкой его хэша), `notebooks/02_robustness_testing.ipynb` (уверенности, вариации текста и p95 inference).
3. **API:** существующий `app/main.py`: `/predict`, `/model-info`, `/healthcheck` не заменены; добавлены `/health/live`, `/health/ready`, `/metrics`; согласованные правила валидации текста для API и Telegram.
4. **Стратегия деплоя:** `docs/DEPLOYMENT_STRATEGY_RU.md`.
5. **Docker:** `Dockerfile`, `compose.yaml`, `.dockerignore`, `requirements-deploy.txt`.
6. **Мониторинг:** `app/metrics.py`, `monitoring/prometheus/*`, `monitoring/grafana/*`.

## Самый быстрый запуск

```bash
cp .env.example .env
# Обязательно смените POSTGRES_PASSWORD и GRAFANA_PASSWORD в .env
docker compose config
docker compose up -d --build
curl -f http://127.0.0.1:8000/health/ready
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"text":"Привет, как дела?"}'
```

Откройте `http://127.0.0.1:3000` (Grafana), `http://127.0.0.1:9090` (Prometheus). Для MLflow: `docker compose --profile mlflow up -d --build mlflow`, открыть `http://127.0.0.1:5000`.

## Эксперименты

```bash
pip install -r requirements-mlops.txt
python -m mlops.import_baseline
# Исходный датасет уже есть в `data/dataset.txt.zip` (в офлайн-архиве проекта):
python -m mlops.audit_dataset --data data/dataset.txt.zip
python -m mlops.train --data data/dataset.txt.zip --output artifacts/candidate_model.joblib
```

Для ноутбука качества: `DATA_PATH=data/dataset.txt.zip jupyter lab notebooks/01_quality_evaluation.ipynb` (SHA-256 распакованного TXT должен совпасть с исходным baseline). Второй ноутбук работает без датасета. MLflow может работать на `file:./mlruns` без сервера, но UI удобнее с tracking server.

## Тестирование

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
python -m pytest -q app/tests
```

Новые тесты независимы от настоящего исходного датасета. Исходные API-тесты включают PostgreSQL через `pgserver-extend`; их запуск требует полной установки зависимостей.

Детальный аудит и **перечень реально выполненных/невыполненных проверок** — `docs/AUDIT_RU.md`.

## Проверено на исходном dataset.txt.zip (20.11.2026)

* Прочитано 248 290 строк. 8 333 многометочные строки исходный парсер пропускает; это **не ошибочные строки** по формату датасета, а ограничение выбранной схемы single-label. Учтите это при оценке покрытия.
* 239 957 однозначно размеченных строк; после нормализации/дедупликации 239 900. Разбиение: 167 930 / 35 985 / 35 985.
* SHA-256 распакованного TXT **совпадает с baseline_metadata.json**. Файл ZIP имеет другой SHA-256; исправлено сравнение для архива.
* Загрузка сохранённой модели и реальные предсказания на тестовой части подтвердили метрики: Accuracy 0.943254, Macro-F1 0.857169, Recall(toxic) 0.880342.
* Исходная модель была обучена на scikit-learn 1.9.1; при проверке с 1.8.0 появились предупреждения совместимости, хотя метрики совпали. Для запуска в Docker закреплена 1.9.1.
* Отчёты и выполненные копии ноутбуков: `reports/`. Три модели повторно обучены и сравнены **без MLflow** в проверочной среде; результаты в `docs/RETRAINING_RESULTS_RU.md`. Фактическую регистрацию run в MLflow нужно выполнить на вашей локальной машине: `bash scripts/local_validate.sh mlflow && bash scripts/local_validate.sh train-all`.

**Персональные данные:** в исходных текстах могут быть номера телефонов и другая персональная информация. `data/` исключена через `.gitignore` и `.dockerignore`; не публикуйте датасет в открытом GitHub без проверки прав и обезличивания. Для публикации проекта используйте архив **без датасета**.

Полный протокол проверки на исходном датасете: [`docs/REAL_DATA_EVALUATION_RU.md`](docs/REAL_DATA_EVALUATION_RU.md).

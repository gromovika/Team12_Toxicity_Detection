# Стратегия деплоя и мониторинга Team12

## 1. Цель, сервисы и выбор архитектуры

Сохраняем исходное приложение `app/main.py` (FastAPI + опциональный Telegram polling) и существующий pipeline `artifacts/baseline_model.joblib`. В Docker-среде используем **внешний PostgreSQL** (контейнер `db`) вместо локального встроенного `pgserver` — это отделяет данные от процесса API. Один процесс `uvicorn --workers 1` важен, если включён Telegram polling, иначе каждый worker попробует запустить бота. MLflow используется для экспериментов и не входит в путь пользовательского запроса. Prometheus собирает агрегированные метрики HTTP и инференса; Grafana отображает их.

Схема: `пользователь / Telegram → FastAPI + ModelService → PostgreSQL`, `Prometheus → API /metrics`, `Grafana → Prometheus`. `MLflow ← offline обучение`, не допускайте произвольный доступ из интернета к UI MLflow.

## 2. Окружения

- **Локальная разработка:** `cd app && python main.py` с `pgserver` или `DB_URL`; `BOT_TOKEN` можно оставить пустым. `pytest app/tests` запускает изолированный PostgreSQL.
- **CI:** сквозная проверка Python-тестов, независимых синтетических ноутбуков, компиляция Python и `docker compose config` в `.github/workflows/ci.yml`. В CI нельзя требовать исходный датасет или реальный Telegram-токен.
- **Staging:** Docker Compose с отдельной БД, отдельными секретами и тестовой моделью/ботом; проверить `/health/ready`, `/predict`, `/metrics`, Grafana, логи.
- **Production (проектная схема):** один экземпляр API + PostgreSQL с persistent volume; Prometheus и Grafana доступны только через localhost или SSH-туннель. Для публичного трафика необходим reverse proxy с HTTPS, доступом по правилам и rate limit. Секреты в переменных окружения/secret manager, не в репозитории.

## 3. Развёртывание (из корня репозитория)

1. Установить Docker Engine с Compose v2. Выполнить `cp .env.example .env`, заменить `POSTGRES_PASSWORD`, `GRAFANA_PASSWORD` на устойчивые случайные значения. В `.env` можно задать `BOT_TOKEN` и `MODEL_VERSION`; никому не передавать `.env`.
2. `docker compose config` — проверка конфигурации.
3. При занятом порту `8000` задайте `API_HOST_PORT=18000` в `.env`, без ручной правки Python или shell-скриптов. `docker compose up -d --build` — запускает `db`, `api`, `prometheus`, `grafana`. При необходимости MLflow: `docker compose --profile mlflow up -d --build mlflow` (UI `http://127.0.0.1:5000`).
4. Проверки: `curl -f http://127.0.0.1:<API_HOST_PORT>/health/live`, `curl -f http://127.0.0.1:<API_HOST_PORT>/health/ready`, `curl -X POST http://127.0.0.1:<API_HOST_PORT>/predict -H 'Content-Type: application/json' -d '{"text":"Привет, как дела?"}'`.
5. Метрики: `curl http://127.0.0.1:<API_HOST_PORT>/metrics`; Prometheus `127.0.0.1:9090` (Targets), Grafana `127.0.0.1:3000` (пользователь `GRAFANA_USER`, пароль `GRAFANA_PASSWORD`).
6. `docker compose logs --tail=100 api db` и `docker compose ps`. Не публикуйте порты наблюдаемости в интернет; всё привязано к `127.0.0.1`.

**Совместимость:** исходный `joblib` сериализован с `scikit-learn 1.9.1`; Docker использует `scikit-learn==1.9.1`. Артефакты `joblib` загружать только из доверенного репозитория. Сборка Docker-образов и запуск локального Compose зафиксированы в [протоколе](LOCAL_VERIFICATION_RU.md).

## 4. Воспроизводимые эксперименты и выпуск новой модели

Модель baseline и отчёты уже присутствуют в `artifacts/`. Для MLflow установите Python-зависимости `pip install -r requirements-mlops.txt`. При работающем сервере задайте `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`; без сервера будет использована локальная папка `mlruns`.

- `python -m mlops.import_baseline` — заносит сохранённые исходные показатели и baseline в MLflow, **без повторного обучения**; run помечается `existing_baseline`, `archived_metadata_not_recomputed`.
- `python -m mlops.train --data data/dataset.txt --output artifacts/candidate_model.joblib` — независимо обучает 3 модели на исходном датасете, записывает параметры и validation-метрики в MLflow, выбирает по validation macro-F1, затем один раз оценивает test. **Не перезаписывает baseline**.
- Для локальной проверки без MLflow: `python -m mlops.train --data data/dataset.txt --no-mlflow`.
- `notebooks/01_quality_evaluation.ipynb` требует **ровно исходный** датасет baseline, верифицирует SHA-256. `notebooks/02_robustness_testing.ipynb` работает на самом артефакте без датасета.

**Quality gate до релиза:** проверить отсутствие пересечения train/valid/test, схему классов и `predict_proba`; сопоставить macro-F1 и toxic recall на неизменных выборках; проверить ошибки и устойчивость, обработку невалидных сообщений, регрессии API и latency. Рекомендуемый ориентир для экспериментов: не хуже baseline (macro-F1 и recall toxic), однако договориться о финальных порогах до повторного тестирования. **Не подбирать алгоритмы/параметры по test.** В текущем `mlops/train.py` SGD hinge завёрнут в `CalibratedClassifierCV(method="sigmoid", cv=3)` и поэтому предоставляет `predict_proba`; это нужно для совместимости с API. Кандидаты не публикуются автоматически.

## 5. Обновление и откат

1. Перед изменением сохранить резервную копию `artifacts/baseline_model.joblib`, `artifacts/baseline_metadata.json`, версию контейнера, сделать backup БД. Проверить наличие файлов у кандидата и соответствие классов.
2. После ручного quality gate скопировать `candidate_model.joblib` в `baseline_model.joblib`, а `candidate_metadata.json` в `baseline_metadata.json`; увеличить `MODEL_VERSION` в корневом `.env`. Это сознательное **ручное** продвижение, не действие CI.
3. `docker compose up -d --build api`, дождаться `/health/ready`, прогнать smoke `/predict` и 5–10 минут следить за ошибками и p95. Предпочтительно публиковать релиз как отдельный Git tag или image tag.
4. При деградации восстановить два baseline-файла из копии, вернуть `MODEL_VERSION`, перезапустить контейнер. Если изменялась схема БД, нужен отдельный проверенный план SQL-миграции/отката. Данные PostgreSQL не откатывать вместе с кодом без оценки потери записей.

Текущий Dockerfile **копирует** `artifacts/` в образ: при замене модели нужна новая сборка image и rolling restart; горячая подмена модели не предполагается.

## 6. Наблюдаемость и реакция

Сервис экспортирует:

- `toxicity_http_requests_total{route,status}` — запросы/коды; включая 422 (не считать ошибками модели).
- `toxicity_http_request_duration_seconds{route}` — HTTP latency.
- `toxicity_predictions_total{source,predicted_class}` — количество успешных предсказаний API и Telegram.
- `toxicity_inference_duration_seconds{source}` — задержка **только** вызова модели, без сети и БД.
- стандартные метрики процесса Python; `/health/live` — процесс отвечает, `/health/ready` — модель загружена и PostgreSQL отвечает.

Alert rules: API недоступен дольше двух минут, более 5% HTTP 5xx за пять минут (10 минут подряд), p95 инференса больше секунды (10 минут подряд). Это начальные пороги, их следует адаптировать к нормальному трафику. Настроенный файл alerts только вычисляет события — **для уведомлений** необходим Alertmanager и отдельная маршрутизация; сейчас оповещение по email/Telegram не настроено.

Измерение дрейфа: распределение предсказанных классов — косвенный сигнал, но не метрика истинного качества. Заполняйте `classification_results.true_class` после ручной разметки и регулярно сверяйте macro-F1/recall по согласованному и достаточному размеченному окну с baseline, без утечки пользовательских текстов в Prometheus.

## 7. Данные и безопасность

База сохраняет **сырые тексты** в `classification_results`; разграничьте доступ, определите сроки хранения, порядок удаления и правила получения согласия. Запрещено экспортировать текст сообщения, токен бота, `DB_URL`, идентификаторы пользователей в labels Prometheus и MLflow. `.env`, `data/`, `mlruns/`, `app/pgdata/` исключены из git. Все порты API/метрик/дашбордов привязаны к localhost, **но это не замена HTTPS, авторизации, firewall или API rate limiting** при публикации. До продакшена добавьте бэкапы и восстановление PostgreSQL с пробным restore.

## 8. Результаты локального запуска

На macOS (20.11.2026) подтверждены сборку Docker, 13+19 тестов с PostgreSQL, MLflow (baseline + 3 candidates + champion), Prometheus UP и Grafana с живыми метриками. Production-развёртывание и доставка alert-уведомлений не проверялись. [Протокол](LOCAL_VERIFICATION_RU.md) и [план мониторинга](MONITORING_PLAN_RU.md).

# Team12 — проверка Docker, PostgreSQL, MLflow и повторного обучения на своём компьютере

Это руководство предназначено для **macOS или Linux**; для Windows используйте WSL2 и Docker Desktop. Рабочие данные БД и секреты не отправляются наружу.

## 0. Требования

- Docker Desktop (macOS/Windows) или Docker Engine + Compose v2 (Linux); запустите Docker Desktop до команд ниже.
- **Python 3.12** (важно: исходная модель сериализована в `scikit-learn==1.9.1`); желательно 4+ ГБ свободной RAM и несколько ГБ диска под MLflow и образы.
- Интернет для первой сборки Docker-образов и установки Python-пакетов.
- **Полный архив**, содержащий `data/dataset.txt.zip`, если необходимо повторное обучение. В публичном архиве исходного датасета нет.

Для macOS с Homebrew (если Python 3.12 не установлен):

```bash
brew install python@3.12
```

Проверка предварительных условий в Terminal:

```bash
docker --version
docker compose version
docker info
python3.12 --version
```

Откройте Terminal в **корне распакованного проекта**, там, где лежит `compose.yaml`:

```bash
cd /ПУТЬ/К/ПАПКЕ_ПРОЕКТА
```

## 1. Python-окружение и безопасные конфигурации

```bash
bash scripts/local_validate.sh setup
```

Команда создаст `.env` со случайными паролями **только если он ещё не существует**, создаст `.venv` на Python 3.12 и установит необходимые зависимости. Никогда не коммитьте `.env` или `data/` в GitHub. `BOT_TOKEN` можно не задавать: API работает без Telegram.

## 2. Полный Docker-стек

**Если на Mac занят порт 8000:** откройте созданный корневой `.env` и выставьте `API_HOST_PORT=18000` (или другой свободный порт). Compose пробросит внешний порт на внутренний `8000`, а скрипт автоматически прочитает значение из `.env`. Больше не нужно делать `sed` по скрипту. **Если на Mac вы ранее вручную меняли `compose.yaml` на `18000`, при переносе правок используйте новое свойство `${API_HOST_PORT:-8000}`, а в `.env` задайте `API_HOST_PORT=18000`.**



```bash
bash scripts/local_validate.sh infra
bash scripts/local_validate.sh status
```

Будут запущены **шесть контейнеров**: основной PostgreSQL (`db`), изолированный тестовый PostgreSQL (`db_test`), FastAPI (`api`), MLflow, Prometheus, Grafana. `db_test` — временная БД, с временной файловой системой tmpfs, доступная только на `127.0.0.1:55432`. Она **никогда не заменяет** основную БД.

Ссылки после старта:

| Назначение | Адрес | Что должно быть |
|---|---|---|
| API Swagger | http://127.0.0.1:<API_HOST_PORT>/docs | Страница FastAPI |
| API readiness | http://127.0.0.1:<API_HOST_PORT>/health/ready | `status: ready` |
| API Prometheus | http://127.0.0.1:<API_HOST_PORT>/metrics | `toxicity_http_requests_total` и другие |
| MLflow | http://127.0.0.1:5000 | Tracking UI |
| Prometheus | http://127.0.0.1:9090/targets | Target API = UP |
| Grafana | http://127.0.0.1:3000 | Вход с паролем из `.env` |

При ошибках:

```bash
docker compose --profile mlflow --profile test ps
docker compose --profile mlflow --profile test logs --tail=120 api db db_test mlflow
```

Если занят **порт API** `8000`, достаточно изменить `API_HOST_PORT` в `.env` (например, `18000`). Другие внешние порты (`5000`, `3000`, `9090`, `55432`) пока задаются в `compose.yaml` напрямую. Если они заняты, нужно отдельно подобрать свободный порт и поправить соответствующие адреса. **Важно:** независимо от порта хоста Prometheus использует `api:8000/metrics` в сети Docker.

## 3. Интеграционные тесты с НАСТОЯЩЕЙ PostgreSQL

```bash
bash scripts/local_validate.sh tests
```

Эта команда запускает отдельно:

1. `python -m pytest -q tests` — тесты обработки данных, обучения, API-метрик, ноутбуков и конфигурации.
2. `python -m pytest -q app/tests` с `TEAM12_TEST_DB_URL` — **все оригинальные API-тесты**, на выделенном PostgreSQL-контейнере. База создаётся сервисом `db_test`, данные тестов хранятся только в нём.

Вы увидите в консоли результаты `passed`. Ни один из этих тестов не должен обращаться к основной БД `db`. Удалить только временную БД, когда закончите: `bash scripts/local_validate.sh clean-test`. **Не выполняйте `docker compose down -v`**, если хотите сохранить основную PostgreSQL, MLflow, Prometheus, Grafana: `-v` удаляет их volume.

## 4. MLflow — регистрация исходного эксперимента

```bash
bash scripts/local_validate.sh mlflow
```

Команда запускает `mlops.import_baseline`, который **не переобучает** модель, а регистрирует оригинальные метрики, параметры и модель в experiment `team12-toxicity`. Затем `mlops.verify_mlflow` проверяет, что run виден по HTTP через `MLflowClient` и сохранены baseline-метрики и исходные отчёты. Результат содержит `baseline_run_id`. Проверьте тот же run в UI: http://127.0.0.1:5000.

Примечание: повторный запуск `mlflow` создаёт ещё один baseline-run. Это допустимо для учебной проверки, но для чистоты эксперимента достаточно одного запуска.

## 5. Повторное обучение с регистрацией в MLflow

**Быстрый реальный эксперимент** (только Naive Bayes):

```bash
bash scripts/local_validate.sh train-fast
bash scripts/local_validate.sh compare
```

**Полный сравнительный эксперимент** (Naive Bayes + Logistic Regression + calibrated SGD SVM):

```bash
bash scripts/local_validate.sh train-all
bash scripts/local_validate.sh compare
```

Каждый алгоритм получает собственный run с validation macro-F1; лучший выбирается **по validation**, затем ровно один раз оценивается на test. Champion записывается в MLflow с `stage=candidate_not_promoted` и в `artifacts/candidate_model.joblib` + `artifacts/candidate_metadata.json`. Рабочие `artifacts/baseline_*` не перезаписываются.

При обучении откройте MLflow UI (experiment `team12-toxicity`) и сравните runs по `macro_f1_val`, а у champion посмотрите `test_macro_f1` и `test_toxic_recall`. Система **не продвигает** кандидата автоматически даже при более высоких метриках: нужна проверка API, данных и совместимости.

## 6. Старые offline-результаты и новый подтверждённый локальный запуск

Вне Docker было успешно выполнено полное обучение трёх алгоритмов на исходном ZIP (под `scikit-learn 1.8.0`, а не pinned 1.9.1); в `docs/RETRAINING_RESULTS_RU.md` сохранены числа, сравнение с baseline и ограничения. **Последующий локальный запуск командой завершён 20.11.2026:** 3 candidate-run + champion сохранены в MLflow; сравнение показывает, что champion и baseline равны по test macro-F1=0,857169 и recall toxic=0,880342. Эти цифры **не** следует смешивать с offline-прогоном под другой версией sklearn. Подробности и IDs: [`docs/LOCAL_VERIFICATION_RU.md`](docs/LOCAL_VERIFICATION_RU.md), [`docs/EXPERIMENTS_RU.md`](docs/EXPERIMENTS_RU.md).

## 7. Зафиксируйте результаты локальной проверки

Проверьте, что одновременно выполняются четыре условия:

- `/health/ready` отвечает 200, `/predict` возвращает корректные классы и вероятности, Prometheus отображает target API как UP;
- `bash scripts/local_validate.sh tests` завершается без ошибок; оригинальные тесты действительно обращаются к `db_test`;
- в MLflow UI есть baseline и новый champion-run с метриками;
- новый `artifacts/candidate_model.joblib` загружается через `ModelService`, а файл `candidate_metadata.json` содержит test macro-F1, toxic recall и хеш датасета.

При любой ошибке пришлите в чат **конкретный текст ошибки из Terminal**, не прикладывайте `.env`, токен Telegram или дамп БД. Для Docker-ошибок достаточно `docker compose --profile mlflow --profile test ps` и соответствующих строк `docker compose ... logs --tail=80 <service>`.

# app — сервис детекции токсичности

Приложение в одном процессе объединяет ML-модель, HTTP API (FastAPI),
Telegram-бота (aiogram) и базу данных (PostgreSQL). Бот и эндпоинты принимают
текст, модель классифицирует его на `neutral` / `offensive` / `toxic`, результат
возвращается пользователю и сохраняется в БД.

Модель — `artifacts/baseline_model.joblib` (Logistic Regression + TF-IDF),
обучена в ноутбуке `Experiments_Data_Retraining 2.ipynb`, test macro-F1 = 0.857.

## Запуск

```bash
cd app
pip install -r requirements.txt
copy .env.example .env   # вставьте токен от @BotFather
python main.py
```

- HTTP API поднимается на `http://127.0.0.1:8000`, документация — `/docs`.
- PostgreSQL поднимается автоматически: встроенный сервер (pgserver), данные
  в папке `pgdata/` (создаётся сама, игнорируется гитом). Docker не нужен.
  Чтобы использовать внешний сервер, задайте `DB_URL` в `.env`.
- Если `BOT_TOKEN` задан, параллельно работает Telegram-бот (polling).
  Если `api.telegram.org` недоступен из вашей сети, задайте `PROXY_URL` или VPN.
- Без `BOT_TOKEN` работает только HTTP API.

## Структура

| Файл | Назначение |
|---|---|
| `main.py` | Приложение: эндпоинты, хендлеры бота, валидация текста |
| `model_service.py` | Загрузка модели, предсказание |
| `db.py` | Модели БД (`model_versions`, `classification_results`) и запись предсказаний |
| `config.py` | Настройки из `.env` |
| `tests/test_api.py` | Тесты (на временном PostgreSQL) |
| `.env.example` | Шаблон настроек (.env игнорируется гитом) |

## База данных

Таблицы по схеме проекта: `model_versions` (версия модели регистрируется
при старте) и `classification_results` (каждое предсказание: текст, класс,
уверенность, источник `telegram`/`api`). Поле `true_class` заполняется при
ручной проверке и служит разметкой для переобучения. Пример работы с БД —
в ноутбуке `database/Toxicity_project_Database.ipynb`.

## Просмотр БД (DBeaver)

Сервер живёт только пока работает приложение; порт подбирается свободный
при каждом запуске. Узнать текущий порт (вторым терминалом из `app/`):

```bash
python -c "import pgserver; print(pgserver.get_server('pgdata').get_uri())"
```

В DBeaver (новое соединение → PostgreSQL): хост `127.0.0.1`, порт из команды
выше, база `postgres`, пользователь `postgres`, пароль пустой. Быстрый просмотр
из терминала без DBeaver: `python view_db.py`. Часовые пояса для embedded-сервера
подкладываются автоматически при старте приложения (пакет `tzdata`).

## Тесты

```bash
cd app
pytest -v
```

19 тестов: эндпоинты, валидация текста, прямой вызов модели, устойчивость,
латентность, запись в БД. Тестовый PostgreSQL поднимается временно,
рабочая база не трогается, бот не стартует.

## Настройки (.env)

- `BOT_TOKEN` — токен Telegram-бота от @BotFather
- `PROXY_URL` — прокси для api.telegram.org (если блокируется)
- `MAX_TEXT_LENGTH` — лимит длины текста, по умолчанию 4096
- `MODEL_PATH`, `MODEL_METADATA_PATH` — пути к модели и метаданным
- `DB_URL` — внешний PostgreSQL; пусто — встроенный сервер
- `PGDATA_DIR` — папка данных встроенного PostgreSQL (по умолчанию `pgdata/`)
- `MODEL_VERSION` — метка версии модели для `model_versions`

## Дополнения MLOps

Новые маршруты: `GET /health/live` (liveness), `GET /health/ready` (модель + PostgreSQL), `GET /metrics` (Prometheus; без сырых сообщений). Для Docker используется внешний PostgreSQL `DB_URL`, один процесс API (из-за Telegram polling) и закреплённая версия `scikit-learn==1.9.1` для исходного baseline. Инструкции: `../README_MLOPS_RU.md`, `../docs/DEPLOYMENT_STRATEGY_RU.md`.

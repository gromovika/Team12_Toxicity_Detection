# app — сервис детекции токсичности

Приложение в одном процессе объединяет ML-модель, HTTP API (FastAPI),
Telegram-бота (aiogram) и базу данных (PostgreSQL). Бот и эндпоинты принимают
текст, модель классифицирует его на `neutral` / `offensive` / `toxic`, результат
возвращается пользователю и сохраняется в БД.

Модель — `artifacts/baseline_model.joblib` (Logistic Regression + TF-IDF),
обучена в ноутбуке `Experiments_Data_Retraining 2.ipynb`, test macro-F1 = 0.857.

## Запуск

Нужны Docker и Docker Compose. Из корня репозитория:

```bash
# полный контур: app + PostgreSQL
docker compose up --build

# или только база, а приложение запускать обычным Python (удобно для разработки)
docker compose up -d db
cd app
pip install -r requirements.txt
copy .env.example .env   # вставьте токен от @BotFather
python main.py
```

- HTTP API: `http://127.0.0.1:8000`, документация — `/docs`.
- PostgreSQL доступен на `localhost:55432` (база `toxicity`, пользователь `postgres`,
  пароль `postgres` или значение переменной `DB_PASSWORD`).
- Если `BOT_TOKEN` задан, параллельно работает Telegram-бот (polling).
  Если `api.telegram.org` недоступен из вашей сети, задайте `PROXY_URL` или VPN.

## Структура

| Файл | Назначение |
|---|---|
| `Dockerfile` | Образ приложения (корень репозитория) |
| `docker-compose.yml` | Связка app + PostgreSQL с volume для данных |
| `app/main.py` | Приложение: эндпоинты, хендлеры бота, валидация текста |
| `app/model_service.py` | Загрузка модели, предсказание |
| `app/db.py` | Модели БД (`model_versions`, `classification_results`) и запись предсказаний |
| `app/config.py` | Настройки из `.env` |
| `app/view_db.py` | Просмотр содержимого БД из терминала |
| `app/tests/test_api.py` | Тесты |
| `app/.env.example` | Шаблон настроек (.env игнорируется гитом) |

## База данных

Таблицы по схеме проекта: `model_versions` (версия модели регистрируется
при старте) и `classification_results` (каждое предсказание: текст, класс,
уверенность, источник `telegram`/`api`). Поле `true_class` заполняется при
ручной проверке и служит разметкой для переобучения. Пример работы с БД —
в ноутбуке `database/Toxicity_project_Database.ipynb`.

Просмотр в DBeaver: новое соединение → PostgreSQL → хост `localhost`,
порт `55432`, база `toxicity`, пользователь `postgres`, пароль `postgres`.

## Тесты

Тестам нужен работающий PostgreSQL (по умолчанию compose-база на localhost:55432;
тестовая база `toxicity_test` создаётся автоматически, рабочие данные не трогаются).
Бот в тестах не стартует.

```bash
docker compose up -d db
cd app
pytest -v
```

Для другого сервера задайте `TEST_DB_URL=postgresql://user:password@host:5432/имя_базы`.

## Настройки (.env)

- `BOT_TOKEN` — токен Telegram-бота от @BotFather
- `PROXY_URL` — прокси для api.telegram.org (если блокируется)
- `MAX_TEXT_LENGTH` — лимит длины текста, по умолчанию 4096
- `DB_URL` — PostgreSQL; по умолчанию compose-база на localhost:55432
- `MODEL_VERSION` — метка версии модели для `model_versions` (менять при переобучении)

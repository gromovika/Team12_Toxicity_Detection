# app — сервис детекции токсичности

Приложение в одном процессе объединяет ML-модель, HTTP API (FastAPI),
Telegram-бота (aiogram) и базу данных (PostgreSQL). Бот и эндпоинты принимают
текст, модель классифицирует его на `neutral` / `offensive` / `toxic`, результат
возвращается пользователю и сохраняется в БД.

Модель — `artifacts/baseline_model.joblib` (Logistic Regression + TF-IDF),
обучена в ноутбуке `Experiments_Data_Retraining 2.ipynb`, test macro-F1 = 0.857.

## Запуск

Нужны Docker и Docker Compose. Все команды — из корня репозитория.

**Вариант А: всё в Docker (приложение + база)**

```bash
docker compose up -d --build
```

**Вариант Б: только база в Docker, приложение обычным Python**

```bash
docker compose up -d db
cd app
pip install -r requirements.txt
copy .env.example .env
python main.py
```

Перед первым запуском откройте `app/.env` и вставьте токен бота от @BotFather
(это единственный обязательный параметр; шаблон с пояснениями — в `.env.example`).

После запуска:

- HTTP API: `http://127.0.0.1:8000`, документация — `/docs`
- PostgreSQL: `localhost:55432`, база `toxicity`, пользователь `postgres`, пароль `postgres`
- бот работает, если задан `BOT_TOKEN`; если `api.telegram.org` блокируется — задайте `PROXY_URL` в `.env` или включите VPN
- остановка: `docker compose down` (данные сохранятся), `docker compose down -v` — с удалением данных

## Мониторинг

Каждый запрос проходит middleware: JSON-запись в лог (путь, метод, статус,
длительность) и накопление счётчиков в памяти. Сводка — эндпоинт `GET /metrics`
(`total_requests`, `total_errors`, `error_rate`, `avg_latency_ms`).

Эскиз дашборда:

| Блок | Метрика | Тип графика |
|---|---|---|
| Инфраструктура | `avg_latency_ms`, `error_rate` | линия во времени за последний час |
| Продукт | число предсказаний по источникам (`telegram`/`api`), доля классов | столбцы по дням |
| Модель | доля проверенных записей (`true_class IS NOT NULL`), точность по версиям (SQL из README БД) | линия по версиям модели |

Сценарии алертов:

| Метрика | Порог | Серьёзность | Действие |
|---|---|---|---|
| error_rate | > 0.05 за 10 минут | Критично | Автоматический откат на предыдущую версию модели |
| avg_latency_ms | > 1000 устойчиво 15 минут | Предупреждение | Проверить загрузку контейнера app |
| /healthcheck | недоступен 1 минуту | Критично | Перезапуск контейнера, разбор логов |
| доля проверенных записей | 0 за сутки | Инфо | Напоминание модераторам о разметке |

## Структура

| Файл | Назначение |
|---|---|
| `Dockerfile` | Образ приложения (в корне репозитория) |
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

Просмотр из терминала (из папки `app`):

```bash
python view_db.py
```

## Тесты

Тестам нужен работающий PostgreSQL (по умолчанию — compose-база). Тестовая база
`toxicity_test` создаётся автоматически, рабочие данные не трогаются.

```bash
docker compose up -d db
cd app
pytest -v
```

Для другого сервера БД задайте переменную окружения `TEST_DB_URL`
вида `postgresql://user:password@host:55432/имя_базы`.

## Настройки (.env)

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `BOT_TOKEN` | токен Telegram-бота от @BotFather | — |
| `PROXY_URL` | прокси для api.telegram.org, если блокируется | — |
| `MAX_TEXT_LENGTH` | лимит длины текста | 4096 |
| `DB_URL` | PostgreSQL | `postgresql://postgres:postgres@localhost:55432/toxicity` |
| `MODEL_VERSION` | метка версии модели в `model_versions` | `baseline` |

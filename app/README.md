# app — сервис детекции токсичности

Приложение в одном процессе объединяет ML-модель, HTTP API (FastAPI) и
Telegram-бота (aiogram). Бот принимает текст сообщения, модель классифицирует
его на `neutral` / `offensive` / `toxic`, результат возвращается пользователю.
Эндпоинты `/predict`, `/healthcheck`, `/model-info` доступны и напрямую, без бота.

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
- Если `BOT_TOKEN` задан, параллельно работает Telegram-бот (polling).
- Если `api.telegram.org` недоступен из вашей сети, задайте `PROXY_URL` в `.env`
  или включите VPN.
- Без `BOT_TOKEN` работает только HTTP API.

## Структура

| Файл | Назначение |
|---|---|
| `main.py` | Приложение: эндпоинты, хендлеры бота, валидация текста |
| `model_service.py` | Загрузка модели, предсказание |
| `config.py` | Настройки из `.env` |
| `tests/test_api.py` | Тесты |
| `.env.example` | Шаблон настроек (.env игнорируется гитом) |

## Тесты

```bash
cd app
pytest -v
```

## Настройки (.env)

- `BOT_TOKEN` — токен Telegram-бота от @BotFather
- `PROXY_URL` — прокси для api.telegram.org (если блокируется)
- `MAX_TEXT_LENGTH` — лимит длины текста, по умолчанию 4096
- `MODEL_PATH`, `MODEL_METADATA_PATH` — пути к модели и метаданным

"""FastAPI + Telegram-бот + ML-модель + PostgreSQL в одном процессе.

Запуск: python main.py (настройки в .env, шаблон — .env.example).
База: PostgreSQL по DB_URL (локально — compose-база на localhost:5432).
"""

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Dict, List

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from config import load_settings
from db import Database
from model_service import ModelService

# Настройки

settings = load_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("app")

# Общий для эндпоинтов и бота объект модели, загружается один раз при старте
service = ModelService(settings.model_path, settings.metadata_path)

# БД создаётся в lifespan: либо по DB_URL, либо встроенный PostgreSQL (pgserver)
db: Database | None = None
model_version_id: int | None = None

# Константы валидации текста

MIN_TEXT_LEN = 2
# Доля «нормальных» символов (буквы/цифры/пробел/пунктуация), ниже — мусор
MIN_MEANINGFUL_RATIO = 0.6

ALLOWED_CHARS = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ\s.,!?;:'\"()\-\n]")
HAS_LETTER = re.compile(r"[a-zA-Zа-яА-ЯёЁ]")

LABEL_NAMES = {
    "neutral": "нейтральный ✅",
    "offensive": "оскорбительный ⚠️",
    "toxic": "токсичный ❌",
}

# Схемы API

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст комментария для классификации")


class PredictResponse(BaseModel):
    label: str
    probabilities: Dict[str, float]
    model: str


class ModelInfo(BaseModel):
    model: str
    classes: List[str]
    test_macro_f1: float
    test_toxic_recall: float
    train_rows: int
    metadata_loaded: bool


class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    error_rate: float
    avg_latency_ms: float


# Валидация текста

class TextVerdict(Enum):
    OK = "ok"
    NOT_TEXT = "not_text"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    GIBBERISH = "gibberish"


def validate_text(message: Message) -> tuple[TextVerdict, str]:
    if message.text is None:
        return TextVerdict.NOT_TEXT, ""
    text = message.text.strip()
    if len(text) < MIN_TEXT_LEN:
        return TextVerdict.TOO_SHORT, text
    if len(text) > settings.max_text_length:
        return TextVerdict.TOO_LONG, text
    if not HAS_LETTER.search(text):
        return TextVerdict.GIBBERISH, text
    meaningful = sum(1 for ch in text if ALLOWED_CHARS.match(ch))
    if meaningful / len(text) < MIN_MEANINGFUL_RATIO:
        return TextVerdict.GIBBERISH, text
    return TextVerdict.OK, text


# Ответы бота

def save_prediction(text: str, result: dict, source: str) -> None:
    """Запись результата классификации; сбой БД не должен ронять ответ."""
    if db is None:
        return
    try:
        db.save_prediction(
            text=text,
            predicted_class=result["label"],
            confidence=max(result["probabilities"].values()),
            model_version_id=model_version_id,
            source=source,
        )
    except Exception:
        log.exception("Не удалось сохранить предсказание в БД")


def build_reply(text: str) -> str:
    """Бот вызывает модель напрямую, минуя HTTP-слой."""
    result = service.predict(text)
    save_prediction(text, result, source="telegram")
    label = result["label"]
    lines = [f"{LABEL_NAMES.get(k, k)}: {v:.1%}" for k, v in result["probabilities"].items()]
    return f"Класс сообщения: {LABEL_NAMES.get(label, label)}\n\n" + "\n".join(lines)


# Хендлеры бота

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Пришли текст комментария — определю, токсичный он, "
        "оскорбительный или нейтральный. Нужен обычный текст."
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        f"Отправь текстовое сообщение от {MIN_TEXT_LEN} до {settings.max_text_length} символов. "
        "Буквы и обычная пунктуация — ок, эмодзи-наборы и мусор — нет."
    )


# Фильтр включает медиа, чтобы на них ответить объяснением, а не промолчать
@router.message(F.text | F.sticker | F.photo | F.video | F.voice | F.document | F.animation | F.video_note)
async def classify(message: Message):
    verdict, text = validate_text(message)
    if verdict == TextVerdict.NOT_TEXT:
        await message.answer("Я работаю только с текстом — пришли текстовое сообщение.")
        return
    if verdict == TextVerdict.TOO_SHORT:
        await message.answer("Сообщение слишком короткое — напиши хотя бы пару символов текста.")
        return
    if verdict == TextVerdict.TOO_LONG:
        await message.answer(f"Слишком длинно (лимит {settings.max_text_length} символов). Сократи и попробуй ещё раз.")
        return
    if verdict == TextVerdict.GIBBERISH:
        await message.answer("Не похоже на осмысленный текст: слишком много спецсимволов или нет букв.")
        return
    try:
        await message.answer(build_reply(text))
    except Exception:
        log.exception("Ошибка классификации")
        await message.answer("Что-то пошло не так при анализе. Попробуй ещё раз.")


# Запуск бота и жизненный цикл приложения

async def run_bot() -> None:
    from aiogram.client.session.aiohttp import AiohttpSession

    session = AiohttpSession(proxy=settings.proxy_url) if settings.proxy_url else None
    bot = Bot(token=settings.bot_token, session=session)
    dp = Dispatcher()
    dp.include_router(router)
    log.info("Бот запущен внутри процесса (polling), жду сообщения")
    await dp.start_polling(bot)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, model_version_id
    service.load()
    log.info("Модель загружена: %s", service.metadata.get("model"))

    db = Database(settings.db_url)
    log.info("БД: PostgreSQL (%s)", settings.db_url.split("@")[-1])

    db.init()
    model_version_id = db.get_or_create_model_version(
        service.metadata.get("model", "unknown"), settings.model_version
    )

    polling_task = None
    if settings.bot_token:
        polling_task = asyncio.create_task(run_bot())
    else:
        log.warning("BOT_TOKEN не задан — только HTTP API, бот не запущен")
    yield
    if polling_task:
        polling_task.cancel()
    db.close()


# Приложение и эндпоинты

app = FastAPI(title="Toxicity Detection API", version="1.0.0", lifespan=lifespan)

# Метрики и логирование запросов: счётчики в памяти + JSON-лог каждого запроса

metrics = {
    "total_requests": 0,
    "total_errors": 0,
    "total_latency_ms": 0.0,
}


@app.middleware("http")
async def logging_and_metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    metrics["total_requests"] += 1
    metrics["total_latency_ms"] += duration_ms
    if response.status_code >= 400:
        metrics["total_errors"] += 1

    log.info(json.dumps({
        "path": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2),
    }, ensure_ascii=False))

    return response


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if len(request.text) > settings.max_text_length:
        raise HTTPException(status_code=422, detail=f"Текст длиннее {settings.max_text_length} символов")
    if not service.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    result = service.predict(request.text)
    save_prediction(request.text, result, source="api")
    return PredictResponse(model=service.metadata.get("model", "unknown"), **result)


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "model_loaded": service.loaded, "bot_running": bool(settings.bot_token)}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    if not service.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    meta = service.metadata
    return ModelInfo(
        model=meta.get("model", "unknown"),
        classes=meta.get("classes", service.classes),
        test_macro_f1=meta.get("test_macro_f1", 0.0),
        test_toxic_recall=meta.get("test_toxic_recall", 0.0),
        train_rows=meta.get("train_rows", 0),
        metadata_loaded=bool(meta),
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    total = metrics["total_requests"]
    return MetricsResponse(
        total_requests=total,
        total_errors=metrics["total_errors"],
        error_rate=round(metrics["total_errors"] / total, 4) if total else 0.0,
        avg_latency_ms=round(metrics["total_latency_ms"] / total, 2) if total else 0.0,
    )


if __name__ == "__main__":
    import os
    import signal
    import threading

    import uvicorn

    def _exit_watchdog(seconds: float):
        # Дедлайн на shutdown: если он завис (aiohttp на Windows), выходим принудительно.
        # Безопасно: данные уже в БД, а контейнерная база живёт независимо от процесса.
        timer = threading.Timer(seconds, lambda: os._exit(0))
        timer.daemon = True
        timer.start()

    def _on_sigint(signum, frame):
        _exit_watchdog(5)
        raise KeyboardInterrupt  # дальше — штатный graceful shutdown uvicorn

    signal.signal(signal.SIGINT, _on_sigint)

    # HOST=0.0.0.0 задаёт compose внутри контейнера
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=8000)

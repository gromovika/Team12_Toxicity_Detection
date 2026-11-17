"""FastAPI + Telegram-бот + ML-модель в одном процессе.

Запуск: python main.py (настройки в .env, шаблон — .env.example).
"""

import asyncio
import logging
import re
from contextlib import asynccontextmanager
from enum import Enum
from typing import Dict, List

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import load_settings
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

def build_reply(text: str) -> str:
    """Бот вызывает модель напрямую, минуя HTTP-слой."""
    result = service.predict(text)
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
    service.load()
    log.info("Модель загружена: %s", service.metadata.get("model"))
    polling_task = None
    if settings.bot_token:
        polling_task = asyncio.create_task(run_bot())
    else:
        log.warning("BOT_TOKEN не задан — только HTTP API, бот не запущен")
    yield
    if polling_task:
        polling_task.cancel()


# Приложение и эндпоинты

app = FastAPI(title="Toxicity Detection API", version="1.0.0", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if len(request.text) > settings.max_text_length:
        raise HTTPException(status_code=422, detail=f"Текст длиннее {settings.max_text_length} символов")
    if not service.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return PredictResponse(model=service.metadata.get("model", "unknown"), **service.predict(request.text))


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

"""Настройки из .env. Новая настройка = поле в Settings + строка в load_settings()."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

# Дефолт — compose-база, замапленная на localhost:55432
# (5432 не используем: там часто сидит локальная установка PostgreSQL)
DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:55432/toxicity"


@dataclass(frozen=True)
class Settings:
    bot_token: str
    proxy_url: str | None
    max_text_length: int
    model_path: Path
    metadata_path: Path
    db_url: str
    model_version: str


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        bot_token=os.getenv("BOT_TOKEN", ""),
        proxy_url=os.getenv("PROXY_URL", "").strip() or None,
        max_text_length=int(os.getenv("MAX_TEXT_LENGTH", "4096")),
        model_path=Path(os.getenv("MODEL_PATH", REPO_ROOT / "artifacts" / "baseline_model.joblib")),
        metadata_path=Path(os.getenv("MODEL_METADATA_PATH", REPO_ROOT / "artifacts" / "baseline_metadata.json")),
        db_url=os.getenv("DB_URL", DEFAULT_DB_URL),
        model_version=os.getenv("MODEL_VERSION", "baseline"),
    )

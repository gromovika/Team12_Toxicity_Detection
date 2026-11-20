import dataclasses
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

import main
from db import Database

# Отдельная тестовая база; по умолчанию — compose-база на localhost:55432.
# Для другого сервера задайте TEST_DB_URL.
TEST_DB_URL = os.environ.get(
    "TEST_DB_URL",
    "postgresql://postgres:postgres@localhost:55432/toxicity_test",
)


def _ensure_database(url: str) -> None:
    """Создаёт тестовую базу, если её нет (нужны права на CREATE DATABASE)."""
    u = make_url(url).set(drivername="postgresql+psycopg2")
    admin = create_engine(u.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": u.database}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{u.database}"'))
    admin.dispose()


@pytest.fixture(scope="module")
def client():
    # Бот в тестах не стартуем; тестовая база отдельная — lifespan подхватит
    # подменённый db_url и не тронет рабочую базу
    main.settings = dataclasses.replace(main.settings, bot_token="", db_url=TEST_DB_URL)
    _ensure_database(TEST_DB_URL)
    main.db = Database(TEST_DB_URL)
    main.db.init()
    main.db.reset()
    with TestClient(main.app) as c:
        yield c
    main.db.close()

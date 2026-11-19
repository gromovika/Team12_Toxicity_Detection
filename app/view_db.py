"""Просмотр содержимого БД: python view_db.py [число записей]"""

import sys

from sqlalchemy import create_engine, text

from config import load_settings
from db import PSYCOPG2_URL_PREFIX


def main():
    settings = load_settings()
    url = settings.db_url
    if url.startswith("postgresql://"):
        url = PSYCOPG2_URL_PREFIX + url.removeprefix("postgresql://")
    engine = create_engine(url)

    with engine.connect() as conn:
        print("== model_versions ==")
        rows = conn.execute(text("SELECT id, version, model_name, created_at FROM model_versions ORDER BY id")).all()
        if not rows:
            print("(пусто)")
        for r in rows:
            print(f"  id={r.id}  version={r.version}  model={r.model_name}  created_at={r.created_at}")

        limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
        print(f"\n== classification_results (последние {limit}) ==")
        rows = conn.execute(
            text(
                "SELECT id, left(text, 40) AS text, predicted_class, true_class, "
                "round(confidence::numeric, 4) AS confidence, source, created_at "
                "FROM classification_results ORDER BY id DESC LIMIT :lim"
            ),
            {"lim": limit},
        ).all()
        if not rows:
            print("(пусто)")
        for r in rows:
            print(f"  id={r.id}  class={r.predicted_class}  true={r.true_class}  conf={r.confidence}  src={r.source}  {r.created_at}  text={r.text!r}")


if __name__ == "__main__":
    main()

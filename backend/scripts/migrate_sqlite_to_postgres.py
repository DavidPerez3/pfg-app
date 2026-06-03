from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from state_store import (
    AppStateStore,
    conversation_events_table,
    feedback_table,
    user_memory_table,
)


def _read_sqlite_rows(sqlite_path: Path, query: str) -> list[dict]:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate the legacy SQLite app-state database into PostgreSQL."
    )
    parser.add_argument("--sqlite-path", required=True, help="Path to the source SQLite file.")
    parser.add_argument(
        "--postgres-url",
        required=True,
        help="Target PostgreSQL SQLAlchemy URL, for example postgresql+psycopg://user:pass@host/db",
    )
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite_path).resolve()
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite source file does not exist: {sqlite_path}")

    target_store = AppStateStore(args.postgres_url)

    feedback_rows = _read_sqlite_rows(sqlite_path, "SELECT * FROM feedback")
    memory_rows = _read_sqlite_rows(sqlite_path, "SELECT * FROM user_memory")
    conversation_rows = _read_sqlite_rows(sqlite_path, "SELECT * FROM conversation_events")

    with target_store.engine.begin() as conn:
        if feedback_rows:
            conn.execute(feedback_table.insert(), feedback_rows)
        if memory_rows:
            conn.execute(user_memory_table.insert(), memory_rows)
        if conversation_rows:
            conn.execute(conversation_events_table.insert(), conversation_rows)

    print(
        "Migration complete:",
        {
            "feedback": len(feedback_rows),
            "user_memory": len(memory_rows),
            "conversation_events": len(conversation_rows),
            "target": args.postgres_url,
        },
    )


if __name__ == "__main__":
    main()

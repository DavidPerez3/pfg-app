from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from state_store import AppStateStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that the PostgreSQL-backed app-state store is reachable."
    )
    parser.add_argument(
        "--database-url",
        required=True,
        help="SQLAlchemy URL, for example postgresql+psycopg://user:pass@host/db",
    )
    args = parser.parse_args()

    store = AppStateStore(args.database_url)
    print(store.db_health())


if __name__ == "__main__":
    main()

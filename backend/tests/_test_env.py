from __future__ import annotations

import os
import tempfile
from pathlib import Path


TEST_DB_PATH = Path(tempfile.gettempdir()) / "pfg-backend-tests.sqlite3"
os.environ.setdefault("APP_STATE_DATABASE_URL", f"sqlite:///{TEST_DB_PATH.as_posix()}")

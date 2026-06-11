from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(APP_ROOT / ".env")


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass(frozen=True)
class Settings:
    frontend_origins: list[str]
    recommender_base_url: str
    ollama_base_url: str
    elasticsearch_url: str
    benchmark_mcp_url: str
    backend_llm_provider: str
    backend_ollama_model: str
    backend_gemini_model: str
    default_dataset: str
    default_rec_model: str
    app_state_database_url: str
    short_term_memory_hours: int
    short_term_memory_max_messages: int
    long_term_memory_index: str
    memory_embedding_model: str


def _default_app_state_database_url() -> str:
    explicit = os.getenv("APP_STATE_DATABASE_URL")
    if explicit:
        return explicit

    legacy_path = os.getenv("APP_STATE_DB_PATH")
    if legacy_path:
        return f"sqlite:///{Path(legacy_path).resolve().as_posix()}"

    return f"sqlite:///{(APP_ROOT / 'app_state.db').resolve().as_posix()}"


settings = Settings(
    frontend_origins=_csv_env("FRONTEND_ORIGINS", "http://localhost:3000"),
    recommender_base_url=os.getenv("RECOMMENDER_BASE_URL", "http://localhost:8001").rstrip("/"),
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
    elasticsearch_url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200").rstrip("/"),
    benchmark_mcp_url=os.getenv("BENCHMARK_MCP_URL", "http://localhost:8010/mcp").rstrip("/"),
    backend_llm_provider=os.getenv("BACKEND_LLM_PROVIDER", "gemini").strip().lower(),
    backend_ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
    backend_gemini_model=os.getenv("BACKEND_GEMINI_MODEL", "gemini-2.5-flash-lite"),
    default_dataset=os.getenv("DEFAULT_DATASET", "movielens"),
    default_rec_model=os.getenv("DEFAULT_REC_MODEL", "mf"),
    app_state_database_url=_default_app_state_database_url(),
    short_term_memory_hours=int(os.getenv("SHORT_TERM_MEMORY_HOURS", "24")),
    short_term_memory_max_messages=int(os.getenv("SHORT_TERM_MEMORY_MAX_MESSAGES", "60")),
    long_term_memory_index=os.getenv("LONG_TERM_MEMORY_INDEX", "pfg_user_memory"),
    memory_embedding_model=os.getenv("MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
)

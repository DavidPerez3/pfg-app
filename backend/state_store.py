from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator

from sqlalchemy import Column, Integer, MetaData, String, Table, Text, create_engine, select, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from config import settings
from shared.contracts import FeedbackRequest, FeedbackSummary, MemoryFact, SessionMemoryMessage


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


metadata = MetaData()

feedback_table = Table(
    "feedback",
    metadata,
    Column("feedback_id", String(64), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("thread_id", String(255), nullable=False),
    Column("rating", Integer, nullable=False),
    Column("comment", Text, nullable=False, default=""),
    Column("message_index", Integer, nullable=True),
    Column("trace_id", String(255), nullable=True),
    Column("created_at", String(64), nullable=False),
)

user_memory_table = Table(
    "user_memory",
    metadata,
    Column("memory_id", String(64), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("fact", Text, nullable=False),
    Column("source", String(64), nullable=False, default="app"),
    Column("created_at", String(64), nullable=False),
)

conversation_events_table = Table(
    "conversation_events",
    metadata,
    Column("event_id", String(64), primary_key=True),
    Column("thread_id", String(255), nullable=False),
    Column("user_id", String(255), nullable=False),
    Column("trace_id", String(255), nullable=True),
    Column("intent", String(128), nullable=True),
    Column("dataset", String(128), nullable=True),
    Column("rec_model", String(128), nullable=True),
    Column("user_message", Text, nullable=False),
    Column("assistant_message", Text, nullable=False),
    Column("created_at", String(64), nullable=False),
)


class AppStateStore:
    """SQL-backed store for app state.

    The store now targets PostgreSQL as the professional default while still
    allowing a SQLite URL during local migration and development.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._lock = threading.Lock()
        self.engine = self._create_engine(database_url)
        self._init_schema()

    def _create_engine(self, database_url: str) -> Engine:
        connect_args: dict[str, object] = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        return create_engine(
            database_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

    @contextmanager
    def _connect(self) -> Iterator[Connection]:
        with self.engine.begin() as conn:
            yield conn

    def _init_schema(self) -> None:
        with self._lock:
            metadata.create_all(self.engine)

    def db_health(self) -> dict[str, str]:
        engine_name = self.engine.url.get_backend_name()
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {
                "status": "ok",
                "engine": engine_name,
                "database_url": self.engine.url.render_as_string(hide_password=True),
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "status": f"error: {exc}",
                "engine": engine_name,
                "database_url": self.engine.url.render_as_string(hide_password=True),
            }

    def store_feedback(self, payload: FeedbackRequest) -> str:
        feedback_id = f"fb-{uuid.uuid4().hex[:12]}"
        row = {
            "feedback_id": feedback_id,
            "user_id": payload.user_id,
            "thread_id": payload.thread_id,
            "rating": payload.rating,
            "comment": payload.comment,
            "message_index": payload.message_index,
            "trace_id": payload.trace_id,
            "created_at": _utc_now(),
        }
        with self._lock, self._connect() as conn:
            conn.execute(feedback_table.insert().values(**row))
        return feedback_id

    def feedback_summary(self) -> FeedbackSummary:
        with self.engine.connect() as conn:
            rows = conn.execute(select(feedback_table.c.rating)).all()
        ratings = [int(row[0]) for row in rows]
        if not ratings:
            return FeedbackSummary(count=0, average_rating=None, distribution={})
        return FeedbackSummary(
            count=len(ratings),
            average_rating=round(sum(ratings) / len(ratings), 2),
            distribution={str(i): ratings.count(i) for i in range(1, 6)},
        )

    def get_latest_feedback(self, thread_id: str) -> dict[str, object] | None:
        query = (
            select(
                feedback_table.c.rating,
                feedback_table.c.comment,
                feedback_table.c.message_index,
                feedback_table.c.trace_id,
                feedback_table.c.created_at,
            )
            .where(feedback_table.c.thread_id == thread_id)
            .order_by(feedback_table.c.created_at.desc())
            .limit(1)
        )
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().first()
        return dict(row) if row else None

    def append_memory_fact(self, user_id: str, fact: str, source: str = "app") -> tuple[str, str]:
        memory_id = f"mem-{uuid.uuid4().hex[:12]}"
        created_at = _utc_now()
        row = {
            "memory_id": memory_id,
            "user_id": user_id,
            "fact": fact.strip(),
            "source": source,
            "created_at": created_at,
        }
        with self._lock, self._connect() as conn:
            conn.execute(user_memory_table.insert().values(**row))
        return memory_id, created_at

    def get_memory(self, user_id: str) -> list[MemoryFact]:
        query = (
            select(
                user_memory_table.c.fact,
                user_memory_table.c.source,
                user_memory_table.c.created_at,
            )
            .where(user_memory_table.c.user_id == user_id)
            .order_by(user_memory_table.c.created_at.desc())
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
        return [MemoryFact(**dict(row)) for row in rows]

    def delete_memory(self, user_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(user_memory_table.delete().where(user_memory_table.c.user_id == user_id))

    def record_conversation_event(
        self,
        *,
        thread_id: str,
        user_id: str,
        trace_id: str | None,
        intent: str | None,
        dataset: str | None,
        rec_model: str | None,
        user_message: str,
        assistant_message: str,
    ) -> str:
        event_id = f"evt-{uuid.uuid4().hex[:12]}"
        row = {
            "event_id": event_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "trace_id": trace_id,
            "intent": intent,
            "dataset": dataset,
            "rec_model": rec_model,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "created_at": _utc_now(),
        }
        with self._lock, self._connect() as conn:
            conn.execute(conversation_events_table.insert().values(**row))
        return event_id

    def get_latest_recommendation_event(self, thread_id: str) -> dict[str, object] | None:
        query = (
            select(
                conversation_events_table.c.intent,
                conversation_events_table.c.dataset,
                conversation_events_table.c.rec_model,
                conversation_events_table.c.user_message,
                conversation_events_table.c.assistant_message,
                conversation_events_table.c.trace_id,
                conversation_events_table.c.created_at,
            )
            .where(conversation_events_table.c.thread_id == thread_id)
            .where(conversation_events_table.c.intent == "user_recommendation")
            .order_by(conversation_events_table.c.created_at.desc())
            .limit(1)
        )
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().first()
        return dict(row) if row else None

    def get_session_messages(
        self,
        thread_id: str,
        *,
        within_hours: int = 24,
        limit_messages: int = 60,
    ) -> list[SessionMemoryMessage]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(within_hours, 1))
        query = (
            select(
                conversation_events_table.c.user_message,
                conversation_events_table.c.assistant_message,
                conversation_events_table.c.created_at,
            )
            .where(conversation_events_table.c.thread_id == thread_id)
            .order_by(conversation_events_table.c.created_at.asc())
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()

        messages: list[SessionMemoryMessage] = []
        for row in rows:
            created_at = str(row["created_at"])
            try:
                created_dt = _parse_utc(created_at)
            except Exception:
                continue
            if created_dt < cutoff:
                continue
            user_message = str(row["user_message"] or "").strip()
            assistant_message = str(row["assistant_message"] or "").strip()
            if user_message:
                messages.append(
                    SessionMemoryMessage(role="user", content=user_message, created_at=created_at)
                )
            if assistant_message:
                messages.append(
                    SessionMemoryMessage(
                        role="assistant",
                        content=assistant_message,
                        created_at=created_at,
                    )
                )

        if limit_messages > 0:
            return messages[-limit_messages:]
        return messages

    def delete_session_history(self, thread_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                conversation_events_table.delete().where(
                    conversation_events_table.c.thread_id == thread_id
                )
            )

    def delete_thread(self, thread_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(feedback_table.delete().where(feedback_table.c.thread_id == thread_id))
            conn.execute(
                conversation_events_table.delete().where(
                    conversation_events_table.c.thread_id == thread_id
                )
            )

    def clear_session(self, thread_id: str) -> None:
        self.delete_session_history(thread_id)


store = AppStateStore(settings.app_state_database_url)

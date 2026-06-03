from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Iterable

import numpy as np

from config import settings
from shared.contracts import MemoryFact, SessionMemoryMessage
from state_store import store
from text_memory_utils import extract_candidate_facts, normalize_fact


log = logging.getLogger(__name__)


class MemoryManager:
    """Two-layer memory manager for the application layer.

    - Short-term memory: canonical backend session history derived from
      conversation events and bounded by a time window.
    - Long-term memory: user preference facts stored in the application SQL
      store and indexed in Elasticsearch for semantic retrieval.
    """

    def __init__(self) -> None:
        self.short_term_hours = settings.short_term_memory_hours
        self.short_term_max_messages = settings.short_term_memory_max_messages
        self.long_term_index = settings.long_term_memory_index
        self.embedding_model_name = settings.memory_embedding_model
        self.es_url = settings.elasticsearch_url

    @lru_cache(maxsize=1)
    def _get_elasticsearch_client(self):
        try:
            from elasticsearch import Elasticsearch
        except Exception as exc:  # pragma: no cover - dependency managed by env
            raise RuntimeError(
                "elasticsearch client is not available in the backend environment."
            ) from exc
        return Elasticsearch(self.es_url, request_timeout=30)

    @lru_cache(maxsize=1)
    def _get_embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - dependency managed by env
            raise RuntimeError(
                "sentence-transformers is not available in the backend environment."
            ) from exc
        return SentenceTransformer(self.embedding_model_name)

    def _ensure_long_term_index(self) -> None:
        es = self._get_elasticsearch_client()
        if es.indices.exists(index=self.long_term_index):
            return
        dims = len(
            np.asarray(
                self._get_embedding_model().encode("memory health probe", show_progress_bar=False),
                dtype=np.float32,
            )
        )
        es.indices.create(
            index=self.long_term_index,
            body={
                "mappings": {
                    "properties": {
                        "memory_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "fact": {"type": "text"},
                        "source": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": dims,
                            "index": True,
                            "similarity": "cosine",
                        },
                    }
                },
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            },
        )

    def short_term_health(self) -> dict[str, str]:
        try:
            db_health = store.db_health()
            return {
                "status": "ok",
                "engine": db_health.get("engine", "unknown"),
                "window_hours": str(self.short_term_hours),
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {"status": f"error: {exc}", "engine": "unknown"}

    def long_term_health(self) -> dict[str, str]:
        try:
            es = self._get_elasticsearch_client()
            index_exists = es.indices.exists(index=self.long_term_index)
            return {
                "status": "ok" if index_exists else "index-missing",
                "engine": "elasticsearch",
                "index": self.long_term_index,
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "status": f"error: {exc}",
                "engine": "elasticsearch",
                "index": self.long_term_index,
            }

    def get_session(self, thread_id: str) -> list[SessionMemoryMessage]:
        return store.get_session_messages(
            thread_id,
            within_hours=self.short_term_hours,
            limit_messages=self.short_term_max_messages,
        )

    def clear_session(self, thread_id: str) -> None:
        store.delete_session_history(thread_id)

    def get_all_facts(self, user_id: str) -> list[MemoryFact]:
        return store.get_memory(user_id)

    def delete_user_memory(self, user_id: str) -> None:
        store.delete_memory(user_id)
        try:
            es = self._get_elasticsearch_client()
            if es.indices.exists(index=self.long_term_index):
                es.delete_by_query(
                    index=self.long_term_index,
                    body={"query": {"term": {"user_id": user_id}}},
                    conflicts="proceed",
                    refresh=True,
                )
        except Exception as exc:
            log.warning("memory.delete_user_memory_es_failed", extra={"error": str(exc)})

    def _normalize_fact(self, fact: str) -> str:
        return normalize_fact(fact)

    def store_facts(self, user_id: str, facts: Iterable[str], source: str = "chat") -> list[str]:
        normalized_facts = [self._normalize_fact(fact) for fact in facts if self._normalize_fact(fact)]
        if not normalized_facts:
            return []

        existing = {self._normalize_fact(fact.fact) for fact in store.get_memory(user_id)}
        new_facts = [fact for fact in normalized_facts if fact not in existing]
        if not new_facts:
            return []

        self._ensure_long_term_index()
        embedder = self._get_embedding_model()
        es = self._get_elasticsearch_client()
        vectors = embedder.encode(new_facts, show_progress_bar=False)
        stored: list[str] = []

        for fact, vector in zip(new_facts, vectors):
            memory_id, created_at = store.append_memory_fact(user_id, fact, source)
            es.index(
                index=self.long_term_index,
                id=memory_id,
                document={
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "fact": fact,
                    "source": source,
                    "created_at": created_at,
                    "embedding": np.asarray(vector, dtype=np.float32).tolist(),
                },
                refresh="wait_for",
            )
            stored.append(fact)
        return stored

    def retrieve_relevant(self, user_id: str, query: str, top_k: int = 5) -> list[str]:
        query = (query or "").strip()
        if not query:
            return []

        try:
            es = self._get_elasticsearch_client()
            if not es.indices.exists(index=self.long_term_index):
                return self._fallback_lexical_retrieve(user_id, query, top_k)
            embedder = self._get_embedding_model()
            vector = np.asarray(
                embedder.encode(query, show_progress_bar=False),
                dtype=np.float32,
            )
            response = es.search(
                index=self.long_term_index,
                body={
                    "knn": {
                        "field": "embedding",
                        "query_vector": vector.tolist(),
                        "k": top_k,
                        "num_candidates": max(top_k * 5, 20),
                        "filter": {"term": {"user_id": user_id}},
                    },
                    "_source": ["fact"],
                    "size": top_k,
                },
            )
            facts: list[str] = []
            seen: set[str] = set()
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                fact = self._normalize_fact(str(source.get("fact", "")))
                if not fact or fact in seen:
                    continue
                seen.add(fact)
                facts.append(fact)
            if facts:
                return facts
        except Exception as exc:
            log.warning("memory.semantic_retrieve_failed", extra={"error": str(exc)})

        return self._fallback_lexical_retrieve(user_id, query, top_k)

    def _fallback_lexical_retrieve(self, user_id: str, query: str, top_k: int) -> list[str]:
        query_tokens = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        if not query_tokens:
            return []
        scored: list[tuple[float, str]] = []
        for fact in store.get_memory(user_id):
            fact_text = self._normalize_fact(fact.fact)
            fact_tokens = set(re.findall(r"[a-zA-Z0-9_]+", fact_text.lower()))
            overlap = len(query_tokens & fact_tokens)
            if overlap == 0:
                continue
            scored.append((overlap / max(len(query_tokens), 1), fact_text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [fact for _, fact in scored[:top_k]]

    def extract_candidate_facts(self, text: str) -> list[str]:
        return extract_candidate_facts(text)


memory_manager = MemoryManager()

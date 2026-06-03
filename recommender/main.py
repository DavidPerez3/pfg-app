"""
Recommender Microservice
========================
FastAPI service with model-specific endpoints.

Endpoints:
  GET  /health
  POST /search          →  Elasticsearch-backed entity lookup with Parquet fallback
  POST /mf              →  Matrix Factorization user recommendations
  POST /mf/similar      →  MF item-to-item similarity
  POST /two_tower       →  Two-Tower recommendations          (future)
  POST /two_tower/similar  →  Two-Tower item similarity       (future)
  POST /sasrec          →  SASRec sequential recommendations  (future)

Usage (run from pfg-app/recommender/):
  uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import logging
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger("recommender")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

app = FastAPI(title="PFG Recommender Microservice", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
API_V1_PREFIX = "/api/v1"

# ── Paths ─────────────────────────────────────────────────────────────────────
# pfg-app/recommender/main.py  →  parents[2] = pfg/
REPO_ROOT  = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from shared.contracts import (  # noqa: E402
    DatasetUserOption,
    DatasetUsersResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendedItem,
    SearchRequest,
    SearchResponse,
    SimilarRequest,
    SimilarResponse,
)

MODELS_ROOT = REPO_ROOT / "pfg-models" / "weights"
DATA_ROOT   = REPO_ROOT / "pfg-models" / "data" / "processed"

SUPPORTED_DATASETS = {"movielens", "amazon_electronics", "yelp", "lastfm", "foursquare"}
MODEL_SLUG_TO_KEY = {
    "matrix-factorization": "matrix_factorization",
    "mf": "matrix_factorization",
    "two-tower": "two_tower",
    "two-tower-wide-deep": "two_tower_wide_deep",
    "sasrec": "sasrec",
    "llm-rag": "llm_rag",
}

MODEL_KEY_TO_LABEL = {
    "matrix_factorization": "Matrix Factorization",
    "two_tower": "Two-Tower",
    "two_tower_wide_deep": "Two-Tower + Wide&Deep",
    "sasrec": "SASRec",
    "llm_rag": "LLM + RAG",
}

GENRE_ALIASES = {
    "comedy": ["comedy", "comedies", "funny", "humor"],
    "drama": ["drama", "dramatic"],
    "action": ["action"],
    "romance": ["romance", "romantic", "love"],
    "horror": ["horror", "scary", "terror"],
    "thriller": ["thriller", "suspense"],
    "sci-fi": ["science fiction", "sci fi", "sci-fi", "scifi", "cyberpunk", "space"],
    "animation": ["animation", "animated", "cartoon"],
    "children": ["children", "kids", "family"],
    "musical": ["musical", "music"],
    "adventure": ["adventure"],
    "crime": ["crime", "detective", "noir"],
    "documentary": ["documentary", "doc"],
    "western": ["western"],
    "war": ["war"],
}


def _resolve_model_key(model_slug: str) -> str:
    model_key = MODEL_SLUG_TO_KEY.get(model_slug)
    if not model_key:
        raise HTTPException(404, f"Unknown recommender model '{model_slug}'.")
    return model_key

def _kv(**kwargs: object) -> str:
    return " ".join(f"{k}={v!r}" for k, v in kwargs.items() if v is not None)

@contextmanager
def _timed_span(trace_id: str | None, stage: str, **kwargs: object):
    start = time.perf_counter()
    log.info("[TRACE][START] trace_id=%r stage=%r %s", trace_id, stage, _kv(**kwargs))
    try:
        yield
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        log.info("[TRACE][END] trace_id=%r stage=%r elapsed_ms=%s", trace_id, stage, elapsed_ms)



# ── Cached loaders ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=16)
def _load_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


@lru_cache(maxsize=4)
def _get_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "torch is required to serve Two-Tower, Two-Tower + Wide&Deep, and SASRec weights."
        ) from exc
    return torch


@lru_cache(maxsize=16)
def _load_torch_state_dict(path: str) -> dict:
    torch = _get_torch()
    return torch.load(path, map_location="cpu")


@lru_cache(maxsize=8)
def _load_items(dataset: str) -> pd.DataFrame:
    path = DATA_ROOT / dataset / "items.parquet"
    if not path.exists():
        raise FileNotFoundError(f"items.parquet not found for dataset '{dataset}': {path}")
    return pd.read_parquet(path).reset_index(drop=True)


@lru_cache(maxsize=8)
def _load_interactions(dataset: str) -> pd.DataFrame:
    path = DATA_ROOT / dataset / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"interactions.parquet not found for dataset '{dataset}': {path}")
    return pd.read_parquet(path, columns=["user_id", "item_id", "rating"])


@lru_cache(maxsize=8)
def _load_interactions_with_timestamp(dataset: str) -> pd.DataFrame:
    path = DATA_ROOT / dataset / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"interactions.parquet not found for dataset '{dataset}': {path}")
    try:
        return pd.read_parquet(path, columns=["user_id", "item_id", "rating", "timestamp"])
    except Exception:
        df = pd.read_parquet(path, columns=["user_id", "item_id", "rating"])
        df["timestamp"] = np.arange(len(df), dtype=np.int64)
        return df


@lru_cache(maxsize=2)
def _get_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "sentence-transformers is not available in the recommender environment."
        ) from exc
    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def _get_elasticsearch_client():
    try:
        from elasticsearch import Elasticsearch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "elasticsearch client is not available in the recommender environment."
        ) from exc

    es_host = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200").rstrip("/")
    return Elasticsearch(es_host, request_timeout=30)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[1]


def _find_optional_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_item_id_col(df: pd.DataFrame) -> str:
    for column in ["item_id", "id", "business_id", "track_id"]:
        if column in df.columns:
            return column
    return df.columns[0]


@lru_cache(maxsize=8)
def _dataset_user_index_map(dataset: str) -> dict[str, int]:
    interactions = _load_interactions(dataset)
    user_values = sorted(interactions["user_id"].unique().tolist())
    return {str(user_id): idx for idx, user_id in enumerate(user_values)}


@lru_cache(maxsize=8)
def _dataset_item_index_map(dataset: str) -> dict[str, int]:
    interactions = _load_interactions(dataset)
    item_values = sorted(interactions["item_id"].unique().tolist())
    return {str(item_id): idx for idx, item_id in enumerate(item_values)}


@lru_cache(maxsize=8)
def _dataset_item_ids(dataset: str) -> tuple[str, ...]:
    interactions = _load_interactions(dataset)
    item_values = sorted(interactions["item_id"].unique().tolist())
    return tuple(str(item_id) for item_id in item_values)


@lru_cache(maxsize=512)
def _dataset_seen_item_indices(dataset: str, dataset_user_id: str) -> tuple[int, ...]:
    interactions = _load_interactions(dataset)
    item_map = _dataset_item_index_map(dataset)
    user_rows = interactions[interactions["user_id"].astype(str) == str(dataset_user_id)]
    if user_rows.empty:
        return ()
    seen = {
        item_map[str(item_id)]
        for item_id in user_rows["item_id"].tolist()
        if str(item_id) in item_map
    }
    return tuple(sorted(seen))


@lru_cache(maxsize=512)
def _dataset_user_sequence_indices(dataset: str, dataset_user_id: str) -> tuple[int, ...]:
    interactions = _load_interactions_with_timestamp(dataset)
    item_map = _dataset_item_index_map(dataset)
    user_rows = interactions[interactions["user_id"].astype(str) == str(dataset_user_id)]
    if user_rows.empty:
        return ()
    ordered = user_rows.sort_values("timestamp")
    sequence = [
        item_map[str(item_id)]
        for item_id in ordered["item_id"].tolist()
        if str(item_id) in item_map
    ]
    return tuple(sequence)


@lru_cache(maxsize=512)
def _dataset_seen_item_ids(dataset: str, dataset_user_id: str) -> tuple[str, ...]:
    interactions = _load_interactions(dataset)
    user_rows = interactions[interactions["user_id"].astype(str) == str(dataset_user_id)]
    if user_rows.empty:
        return ()
    seen: list[str] = []
    seen_set: set[str] = set()
    for item_id in user_rows["item_id"].tolist():
        item_id_str = str(item_id)
        if item_id_str in seen_set:
            continue
        seen_set.add(item_id_str)
        seen.append(item_id_str)
    return tuple(seen)


@lru_cache(maxsize=512)
def _dataset_user_recent_item_ids(dataset: str, dataset_user_id: str, limit: int = 8) -> tuple[str, ...]:
    interactions = _load_interactions_with_timestamp(dataset)
    user_rows = interactions[interactions["user_id"].astype(str) == str(dataset_user_id)]
    if user_rows.empty:
        return ()
    ordered = user_rows.sort_values("timestamp", ascending=False)
    recent: list[str] = []
    recent_set: set[str] = set()
    for item_id in ordered["item_id"].tolist():
        item_id_str = str(item_id)
        if item_id_str in recent_set:
            continue
        recent_set.add(item_id_str)
        recent.append(item_id_str)
        if len(recent) >= limit:
            break
    return tuple(recent)


@lru_cache(maxsize=8)
def _dataset_user_options(dataset: str) -> tuple[list[DatasetUserOption], int]:
    interactions = _load_interactions(dataset)
    counts = (
        interactions.groupby("user_id")
        .size()
        .sort_values(ascending=False)
    )
    options = [
        DatasetUserOption(user_id=str(user_id), interaction_count=int(count))
        for user_id, count in counts.items()
    ]
    return options, len(options)


def _clip_valid_indices(indices: set[int] | list[int], upper_bound: int) -> list[int]:
    return sorted({idx for idx in indices if 0 <= idx < upper_bound})


@lru_cache(maxsize=8)
def _dataset_item_text_by_id(dataset: str) -> dict[str, tuple[str, str]]:
    items_df = _load_items(dataset)
    item_id_col = _get_item_id_col(items_df)
    title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
    genre_col = _find_optional_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])
    text_map: dict[str, tuple[str, str]] = {}
    for _, row in items_df.iterrows():
        item_id = str(row.get(item_id_col, ""))
        text_map[item_id] = (
            _normalize_text(str(row.get(title_col, ""))),
            _normalize_text(str(row.get(genre_col, ""))) if genre_col else "",
        )
    return text_map


@lru_cache(maxsize=8)
def _dataset_item_display_by_id(dataset: str) -> dict[str, tuple[str, str]]:
    items_df = _load_items(dataset)
    item_id_col = _get_item_id_col(items_df)
    title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
    genre_col = _find_optional_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])
    display_map: dict[str, tuple[str, str]] = {}
    for _, row in items_df.iterrows():
        item_id = str(row.get(item_id_col, ""))
        title = str(row.get(title_col, "")).strip() or item_id or "Unknown item"
        genres = str(row.get(genre_col, "")).strip() if genre_col else ""
        display_map[item_id] = (title, genres)
    return display_map


@lru_cache(maxsize=512)
def _dataset_user_profile_context(dataset: str, dataset_user_id: str) -> str:
    recent_item_ids = _dataset_user_recent_item_ids(dataset, dataset_user_id, 8)
    if not recent_item_ids:
        return ""

    display_map = _dataset_item_display_by_id(dataset)
    recent_descriptions: list[str] = []
    genre_counter: Counter[str] = Counter()

    for item_id in recent_item_ids:
        title, genres = display_map.get(item_id, (item_id, ""))
        if genres:
            split = [genre.strip() for genre in genres.split("|") if genre.strip()]
            genre_counter.update(split[:3])
            recent_descriptions.append(f"{title} ({' | '.join(split[:3])})")
        else:
            recent_descriptions.append(title)

    parts = [f"Recent history: {', '.join(recent_descriptions[:6])}."]
    if genre_counter:
        top_genres = ", ".join(genre for genre, _ in genre_counter.most_common(4))
        parts.append(f"Frequent themes: {top_genres}.")
    return " ".join(parts)


def _popular_item_ids(dataset: str, n: int) -> list[str]:
    interactions = _load_interactions(dataset)
    pop = interactions.groupby("item_id").size().sort_values(ascending=False)
    return [str(item_id) for item_id in pop.index[:n]]


def _indices_to_item_ids(dataset: str, indices: list[int]) -> list[str]:
    item_ids = _dataset_item_ids(dataset)
    valid = _clip_valid_indices(indices, len(item_ids))
    return [item_ids[idx] for idx in valid]


def _filter_item_ids_by_genres(
    dataset: str,
    candidate_item_ids: list[str],
    include_genres: set[str],
    exclude_genres: set[str] | None = None,
) -> list[str]:
    exclude_genres = exclude_genres or set()
    if not include_genres and not exclude_genres:
        return candidate_item_ids
    text_map = _dataset_item_text_by_id(dataset)
    filtered: list[str] = []
    for item_id in candidate_item_ids:
        _, item_genres = text_map.get(str(item_id), ("", ""))
        if exclude_genres and any(genre in item_genres for genre in exclude_genres):
            continue
        if not include_genres or any(genre in item_genres for genre in include_genres):
            filtered.append(str(item_id))
    return filtered


def _rerank_item_ids_by_prompt(dataset: str, candidate_item_ids: list[str], prompt: str, top_k: int) -> list[str]:
    tokens = _prompt_tokens(prompt)
    include_genres, exclude_genres = _genre_constraints_from_prompt(prompt)
    modifiers = _request_modifiers(prompt)
    if not tokens and not include_genres and not exclude_genres and not any(modifiers.values()):
        return candidate_item_ids[:top_k]

    text_map = _dataset_item_text_by_id(dataset)
    display_map = _dataset_item_display_by_id(dataset)
    popularity_map = _dataset_item_popularity_map(dataset)
    scored: list[tuple[float, str]] = []
    for rank, item_id in enumerate(candidate_item_ids):
        title, genres = text_map.get(str(item_id), ("", ""))
        display_title, _ = display_map.get(str(item_id), (str(item_id), ""))
        year_match = re.search(r"\((19|20)\d{2}\)", display_title)
        year_score = 0.0
        if year_match:
            year_score = min(max((float(year_match.group()[1:5]) - 1950.0) / 100.0, 0.0), 1.0)
        score = _candidate_score(
            base_rank=rank,
            candidate_count=len(candidate_item_ids),
            title_text=title,
            genre_text=genres,
            popularity_score=popularity_map.get(str(item_id), 0.0),
            year_score=year_score,
            tokens=tokens,
            include_genres=include_genres,
            exclude_genres=exclude_genres,
            modifiers=modifiers,
        )
        scored.append((score, str(item_id)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item_id for _, item_id in scored[:top_k]]


def _history_profile_vector(
    item_embeddings: np.ndarray,
    item_indices: list[int] | tuple[int, ...],
    *,
    sequential: bool = False,
) -> np.ndarray | None:
    valid = _clip_valid_indices(list(item_indices), len(item_embeddings))
    if not valid:
        return None
    embeddings = item_embeddings[valid]
    if sequential:
        weights = np.linspace(1.0, 2.5, num=len(valid), dtype=np.float32)
    else:
        weights = np.ones(len(valid), dtype=np.float32)
    vector = (embeddings * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-9)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def _cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    q = query / (np.linalg.norm(query) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    scores = (matrix / norms) @ q
    topk = np.argpartition(scores, -k)[-k:]
    return topk[np.argsort(scores[topk])[::-1]].tolist()


def _popularity_fallback(dataset: str, n: int) -> list[int]:
    try:
        interactions = _load_interactions(dataset)
        pop = interactions.groupby("item_id").size().nlargest(n)
        return [int(i) for i in pop.index][:n]
    except Exception:
        return list(range(n))


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", _normalize_text(text))


def _extract_current_request(prompt: str) -> str:
    marker = "Current refinement request:"
    if marker in prompt:
        tail = prompt.split(marker, 1)[1].strip()
        if tail:
            return tail
    return prompt.strip()


def _text_has_negative_genre_signal(text: str, alias: str) -> bool:
    negative_patterns = (
        f"without {alias}",
        f"no {alias}",
        f"not {alias}",
        f"avoid {alias}",
        f"exclude {alias}",
        f"less {alias}",
    )
    return any(pattern in text for pattern in negative_patterns)


def _genre_constraints_from_prompt(prompt: str) -> tuple[set[str], set[str]]:
    text = _normalize_text(_extract_current_request(prompt))
    include_targets: set[str] = set()
    exclude_targets: set[str] = set()
    for canonical, aliases in GENRE_ALIASES.items():
        matched = False
        negative = False
        for alias in aliases:
            alias_text = _normalize_text(alias)
            if alias_text not in text:
                continue
            matched = True
            if _text_has_negative_genre_signal(text, alias_text):
                negative = True
        if not matched:
            continue
        if negative:
            exclude_targets.add(canonical)
        else:
            include_targets.add(canonical)
    include_targets -= exclude_targets
    return include_targets, exclude_targets


def _prompt_tokens(prompt: str) -> list[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "into", "about", "this", "that",
        "want", "would", "like", "please", "recommend", "recommendation",
        "movie", "movies", "film", "films", "show", "shows", "something",
        "previous", "current", "request", "assistant", "summary", "ranking",
        "dataset", "user", "profile", "selected", "treat", "message",
        "refinement", "unrelated", "these", "more",
    }
    tokens = _tokenize_text(_extract_current_request(prompt))
    return [t for t in tokens if len(t) >= 4 and t not in stopwords]


def _genre_targets_from_prompt(prompt: str) -> set[str]:
    include_targets, _ = _genre_constraints_from_prompt(prompt)
    return include_targets


def _request_modifiers(prompt: str) -> dict[str, bool]:
    text = _normalize_text(_extract_current_request(prompt))
    niche = any(token in text for token in ("niche", "obscure", "underrated", "less mainstream", "deeper cut"))
    mainstream = any(token in text for token in ("mainstream", "popular", "broader", "safer")) and not niche
    return {
        "recent": any(token in text for token in ("recent", "newer", "newest", "modern", "latest")),
        "mainstream": mainstream,
        "niche": niche,
        "diverse": any(token in text for token in ("diverse", "varied", "variety", "mixed")),
    }


@lru_cache(maxsize=8)
def _dataset_item_popularity_map(dataset: str) -> dict[str, float]:
    interactions = _load_interactions(dataset)
    counts = interactions.groupby("item_id").size()
    if counts.empty:
        return {}
    max_count = float(counts.max()) or 1.0
    return {str(item_id): float(count) / max_count for item_id, count in counts.items()}


def _item_year_score(row: pd.Series) -> float:
    raw_year = row.get("year")
    if pd.notna(raw_year):
        try:
            year = float(raw_year)
            if year > 1900:
                return min(max((year - 1950.0) / 100.0, 0.0), 1.0)
        except Exception:
            pass
    title = str(row.get("title", row.get("item_name", row.get("name", ""))))
    match = re.search(r"\((19|20)\d{2}\)", title)
    if match:
        year = float(match.group()[1:5])
        return min(max((year - 1950.0) / 100.0, 0.0), 1.0)
    return 0.0


def _genre_list(raw_genres: str) -> list[str]:
    return [genre.strip().lower() for genre in raw_genres.split("|") if genre.strip()]


def _genre_alias_terms(genres: set[str]) -> set[str]:
    terms: set[str] = set()
    for genre in genres:
        terms.add(genre)
        for alias in GENRE_ALIASES.get(genre, []):
            terms.add(_normalize_text(alias))
    return terms


def _candidate_score(
    *,
    base_rank: int,
    candidate_count: int,
    title_text: str,
    genre_text: str,
    popularity_score: float,
    year_score: float,
    tokens: list[str],
    include_genres: set[str],
    exclude_genres: set[str],
    modifiers: dict[str, bool],
) -> float:
    base = max(0.0, 1.0 - base_rank / max(candidate_count, 1))
    title_hits = sum(1 for token in tokens if token in title_text)
    item_genres = _genre_list(genre_text)
    genre_hits = sum(1 for genre in include_genres if genre in item_genres)
    excluded_hits = sum(1 for genre in exclude_genres if genre in item_genres)
    excluded_terms = _genre_alias_terms(exclude_genres)
    unique_genre_count = len(item_genres)

    score = base
    score += 0.9 * title_hits
    score += 2.8 * genre_hits
    score -= 5.0 * excluded_hits
    if excluded_terms and any(term in title_text for term in excluded_terms):
        score -= 4.0

    if include_genres and not genre_hits and genre_text:
        score -= 2.0

    if modifiers["recent"]:
        score += 2.6 * year_score
    if modifiers["mainstream"]:
        score += 1.8 * popularity_score
    if modifiers["niche"]:
        score += 1.8 * (1.0 - popularity_score)
    if modifiers["diverse"]:
        score += 0.7 * min(unique_genre_count, 4)
        score += 0.9 * (1.0 - popularity_score)

    return score


def _filter_candidates_by_genres(
    dataset: str,
    candidate_indices: list[int],
    include_genres: set[str],
    exclude_genres: set[str] | None = None,
) -> list[int]:
    exclude_genres = exclude_genres or set()
    if not include_genres and not exclude_genres:
        return candidate_indices
    items_df = _load_items(dataset)
    genre_col = _find_optional_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])
    if not genre_col:
        return candidate_indices
    filtered: list[int] = []
    for idx in candidate_indices:
        if idx >= len(items_df):
            continue
        row_genres = _normalize_text(str(items_df.iloc[idx].get(genre_col, "")))
        if exclude_genres and any(g in row_genres for g in exclude_genres):
            continue
        if not include_genres or any(g in row_genres for g in include_genres):
            filtered.append(idx)
    return filtered


def _rerank_by_prompt(dataset: str, candidate_indices: list[int], prompt: str, top_k: int) -> list[int]:
    tokens = _prompt_tokens(prompt)
    include_genres, exclude_genres = _genre_constraints_from_prompt(prompt)
    modifiers = _request_modifiers(prompt)
    if not tokens and not include_genres and not exclude_genres and not any(modifiers.values()):
        return candidate_indices[:top_k]

    items_df = _load_items(dataset)
    title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
    item_id_col = _get_item_id_col(items_df)
    genre_col = _find_optional_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])
    popularity_map = _dataset_item_popularity_map(dataset)

    scored: list[tuple[float, int]] = []
    for rank, idx in enumerate(candidate_indices):
        if idx >= len(items_df):
            continue
        row = items_df.iloc[idx]
        title = _normalize_text(str(row.get(title_col, "")))
        genres = _normalize_text(str(row.get(genre_col, ""))) if genre_col else ""
        popularity_score = popularity_map.get(str(row.get(item_id_col, "")), 0.0)
        score = _candidate_score(
            base_rank=rank,
            candidate_count=len(candidate_indices),
            title_text=title,
            genre_text=genres,
            popularity_score=popularity_score,
            year_score=_item_year_score(row),
            tokens=tokens,
            include_genres=include_genres,
            exclude_genres=exclude_genres,
            modifiers=modifiers,
        )
        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [idx for _, idx in scored]
    return reranked[:top_k]


def _build_response(
    user_id: str, dataset: str, model: str,
    item_indices: list[int], cold_start: bool, trace_id: str | None = None,
) -> RecommendResponse:
    items_df = _load_items(dataset)
    title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
    genre_col = _find_optional_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])

    recs = []
    for rank, idx in enumerate(item_indices):
        if idx >= len(items_df):
            continue
        row = items_df.iloc[idx]
        genres = str(row.get(genre_col, "")).strip() if genre_col else ""
        recs.append(RecommendedItem(
            title=str(row.get(title_col, f"Item {idx}")),
            score=round(1.0 - rank / len(item_indices), 3),
            genres=genres,
        ))

    return RecommendResponse(
        user_id=user_id,
        dataset=dataset,
        model=model,
        cold_start=cold_start,
        items=recs,
        trace_id=trace_id,
    )


def _build_response_from_item_ids(
    user_id: str,
    dataset: str,
    model: str,
    item_ids: list[str],
    cold_start: bool,
    trace_id: str | None = None,
    explanation: str | None = None,
) -> RecommendResponse:
    items_df = _load_items(dataset)
    item_id_col = _get_item_id_col(items_df)
    title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
    genre_col = _find_optional_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])

    id_to_row = {str(row[item_id_col]): row for _, row in items_df.iterrows()}

    recs = []
    for rank, item_id in enumerate(item_ids):
        row = id_to_row.get(str(item_id))
        if row is None:
            continue
        genres = str(row.get(genre_col, "")).strip() if genre_col else ""
        recs.append(
            RecommendedItem(
                title=str(row.get(title_col, f"Item {item_id}")),
                score=round(1.0 - rank / max(len(item_ids), 1), 3),
                genres=genres,
            )
        )

    return RecommendResponse(
        user_id=user_id,
        dataset=dataset,
        model=model,
        cold_start=cold_start,
        items=recs,
        trace_id=trace_id,
        explanation=explanation,
    )


def _similar_items_from_item_ids(dataset: str, item_ids: list[str], top_k: int) -> list[RecommendedItem]:
    items_df = _load_items(dataset)
    item_id_col = _get_item_id_col(items_df)
    title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
    genre_col = _find_optional_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])
    id_to_row = {str(row[item_id_col]): row for _, row in items_df.iterrows()}

    items: list[RecommendedItem] = []
    for rank, item_id in enumerate(item_ids):
        row = id_to_row.get(str(item_id))
        if row is None:
            continue
        genres = str(row.get(genre_col, "")).strip() if genre_col else ""
        items.append(
            RecommendedItem(
                title=str(row.get(title_col, f"Item {item_id}")),
                score=round(1.0 - rank / max(top_k, 1), 3),
                genres=genres,
            )
        )
    return items


def _log_recommendation_preview(
    tag: str,
    dataset: str,
    model: str,
    items: list[RecommendedItem],
    cold_start: Optional[bool] = None,
    trace_id: str | None = None,
):
    preview = [it.title for it in items[:5]]
    base = f"{tag} dataset={dataset!r} model={model!r} n_items={len(items)} preview_titles={preview}"
    if cold_start is None:
        log.info("%s trace_id=%r", base, trace_id)
    else:
        log.info("%s cold_start=%s trace_id=%r", base, cold_start, trace_id)


@lru_cache(maxsize=8)
def _two_tower_user_embeddings(dataset: str) -> np.ndarray:
    model_dir = MODELS_ROOT / "two_tower" / dataset
    weights_path = model_dir / "weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Two-Tower weights not found for dataset '{dataset}': {weights_path}")
    state_dict = _load_torch_state_dict(str(weights_path))
    user_emb = state_dict.get("user_emb.weight")
    if user_emb is None:
        raise RuntimeError(f"Two-Tower state_dict for dataset '{dataset}' does not contain 'user_emb.weight'")
    return user_emb.detach().cpu().numpy().astype(np.float32)


def _state_value_to_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32)


@lru_cache(maxsize=8)
def _two_tower_wide_deep_arrays(dataset: str) -> dict[str, np.ndarray]:
    model_dir = MODELS_ROOT / "two_tower_wide_deep" / dataset
    weights_path = model_dir / "weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Two-Tower + Wide&Deep weights not found for dataset '{dataset}': {weights_path}"
        )
    state_dict = _load_torch_state_dict(str(weights_path))
    required = [
        "user_emb.weight",
        "wide.weight",
        "wide.bias",
        "deep.0.weight",
        "deep.0.bias",
        "deep.3.weight",
        "deep.3.bias",
        "deep.6.weight",
        "deep.6.bias",
    ]
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise RuntimeError(
            f"Two-Tower + Wide&Deep state_dict for dataset '{dataset}' is missing keys: {missing}"
        )
    return {key: _state_value_to_numpy(state_dict[key]) for key in required}


def _score_two_tower_wide_deep_candidates(
    dataset: str,
    user_vector: np.ndarray,
    user_idx: int | None,
    candidate_indices: list[int],
) -> list[int]:
    arrays = _two_tower_wide_deep_arrays(dataset)
    user_matrix = arrays["user_emb.weight"]
    if user_idx is not None and 0 <= user_idx < len(user_matrix):
        user_repr = user_matrix[user_idx]
    else:
        user_repr = user_vector
    if user_repr is None:
        return candidate_indices

    item_embeddings = _load_npy(str(MODELS_ROOT / "two_tower_wide_deep" / dataset / "item_embeddings.npy"))
    valid_candidates = _clip_valid_indices(candidate_indices, len(item_embeddings))
    if not valid_candidates:
        return []

    x = np.concatenate(
        [np.repeat(user_repr[None, :], len(valid_candidates), axis=0), item_embeddings[valid_candidates]],
        axis=1,
    )
    wide_scores = (x @ arrays["wide.weight"].T).squeeze(-1) + arrays["wide.bias"].squeeze()

    h1 = np.maximum(0.0, x @ arrays["deep.0.weight"].T + arrays["deep.0.bias"])
    h2 = np.maximum(0.0, h1 @ arrays["deep.3.weight"].T + arrays["deep.3.bias"])
    deep_scores = (h2 @ arrays["deep.6.weight"].T).squeeze(-1) + arrays["deep.6.bias"].squeeze()
    total_scores = wide_scores + deep_scores
    ranked_positions = np.argsort(-total_scores)
    return [valid_candidates[pos] for pos in ranked_positions]


class _SASRecServingModel:
    def __init__(self, dataset: str):
        torch = _get_torch()
        import torch.nn as nn

        state_dict = _load_torch_state_dict(str(MODELS_ROOT / "sasrec" / dataset / "weights.pt"))
        item_weight = state_dict.get("item_emb.weight")
        pos_weight = state_dict.get("pos_emb.weight")
        if item_weight is None or pos_weight is None:
            raise RuntimeError(f"SASRec state_dict for dataset '{dataset}' is missing embedding weights")

        d_model = int(item_weight.shape[1])
        n_items = int(item_weight.shape[0] - 1)
        max_len = int(pos_weight.shape[0])
        layer_ids = {
            int(key.split(".")[2])
            for key in state_dict.keys()
            if key.startswith("transformer.layers.")
        }
        n_layers = max(layer_ids) + 1 if layer_ids else 2
        n_heads = 2 if d_model % 2 == 0 else 1

        class SASRecServing(nn.Module):
            def __init__(self):
                super().__init__()
                self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
                self.pos_emb = nn.Embedding(max_len, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=0.0,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.norm = nn.LayerNorm(d_model)

            def _causal_mask(self, seq_len: int, device):
                return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

            def encode(self, seq):
                batch, length = seq.shape
                pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, length)
                x = self.item_emb(seq) + self.pos_emb(pos)
                key_padding_mask = seq == 0
                causal = self._causal_mask(length, seq.device)
                x = self.transformer(x, mask=causal, src_key_padding_mask=key_padding_mask)
                return self.norm(x)

            def forward(self, seq):
                hidden = self.encode(seq)
                lengths = (seq != 0).sum(dim=1) - 1
                lengths = lengths.clamp(min=0)
                return hidden[torch.arange(hidden.size(0)), lengths]

        self.torch = torch
        self.max_len = max_len
        self.model = SASRecServing()
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def encode_sequence(self, sequence_1_indexed: list[int]) -> np.ndarray:
        seq = sequence_1_indexed[-self.max_len :]
        padded = [0] * (self.max_len - len(seq)) + seq
        with self.torch.no_grad():
            tensor = self.torch.tensor([padded], dtype=self.torch.long)
            encoded = self.model(tensor)[0].detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(encoded)
        if norm > 0:
            encoded = encoded / norm
        return encoded


@lru_cache(maxsize=8)
def _get_sasrec_serving_model(dataset: str) -> _SASRecServingModel:
    weights_path = MODELS_ROOT / "sasrec" / dataset / "weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"SASRec weights not found for dataset '{dataset}': {weights_path}")
    return _SASRecServingModel(dataset)


def _rag_index_name(dataset: str) -> str:
    return f"pfg_{dataset}_items"


def _search_items_from_elasticsearch(req: SearchRequest) -> list[RecommendedItem]:
    query = (req.query or "").strip()
    if not query:
        return []

    es = _get_elasticsearch_client()
    es_index = _rag_index_name(req.dataset)
    if not es.indices.exists(index=es_index):
        raise RuntimeError(
            f"Elasticsearch index '{es_index}' does not exist for dataset '{req.dataset}'."
        )

    display_map = _dataset_item_display_by_id(req.dataset)
    response = es.search(
        index=es_index,
        body={
            "size": req.top_k,
            "_source": [
                "item_id",
                "text_repr",
                "title",
                "genres",
                "artist",
                "track",
                "categories",
                "city",
            ],
            "query": {
                "bool": {
                    "should": [
                        {"term": {"item_id": {"value": query, "boost": 12.0}}},
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "title^6",
                                    "text_repr^5",
                                    "artist^4",
                                    "track^4",
                                    "categories^3",
                                    "city^2",
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "title^6",
                                    "text_repr^5",
                                    "artist^4",
                                    "track^4",
                                    "categories^3",
                                    "city^2",
                                ],
                                "type": "phrase_prefix",
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
        },
    )

    results: list[RecommendedItem] = []
    seen_item_ids: set[str] = set()
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        item_id = str(source.get("item_id", "")).strip()
        if not item_id or item_id in seen_item_ids:
            continue
        seen_item_ids.add(item_id)

        title, genres = display_map.get(item_id, ("", ""))
        if not title:
            title = (
                str(source.get("title", "")).strip()
                or " - ".join(part for part in [source.get("artist"), source.get("track")] if part).strip()
                or str(source.get("text_repr", "")).strip()
                or item_id
            )
        if not genres:
            genres = str(source.get("genres") or source.get("categories") or "").strip()

        results.append(
            RecommendedItem(
                title=title,
                score=round(float(hit.get("_score") or 0.0), 3),
                genres=genres,
            )
        )
        if len(results) >= req.top_k:
            break

    return results


def _search_items_from_parquet(req: SearchRequest) -> list[RecommendedItem]:
    items_df = _load_items(req.dataset)
    title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
    genre_col = _find_optional_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])

    mask = items_df[title_col].astype(str).str.contains(req.query, case=False, na=False)
    matches = items_df[mask].head(req.top_k)

    if matches.empty:
        query_tokens = _prompt_tokens(req.query)
        if query_tokens:
            scored_rows: list[tuple[float, int]] = []
            for idx, row in items_df.iterrows():
                title = str(row.get(title_col, ""))
                title_tokens = set(_tokenize_text(title))
                if not title_tokens:
                    continue
                overlap = sum(1 for t in query_tokens if t in title_tokens)
                if overlap == 0:
                    continue
                score = overlap / max(len(set(query_tokens)), 1)
                scored_rows.append((score, int(idx)))
            scored_rows.sort(key=lambda x: x[0], reverse=True)
            top_idx = [idx for _, idx in scored_rows[: req.top_k]]
            if top_idx:
                matches = items_df.iloc[top_idx]

    return [
        RecommendedItem(
            title=str(row[title_col]),
            score=1.0,
            genres=str(row.get(genre_col, "")).strip() if genre_col else "",
        )
        for _, row in matches.iterrows()
    ]


def _rag_build_query_text(req: RecommendRequest) -> str:
    prompt = _extract_current_request((req.prompt or "").strip())
    dataset_user_id = (req.dataset_user_id or "").strip()
    if dataset_user_id:
        profile_context = _dataset_user_profile_context(req.dataset, dataset_user_id)
        if profile_context:
            prompt_text = prompt or "Recommend items that fit this dataset-user profile."
            return (
                f"User request: {prompt_text}\n"
                f"Dataset-user profile: {profile_context}"
            )
    if prompt:
        return prompt
    return f"recommend items for user {req.user_id}"


def _rag_knn_candidates_from_vector(
    dataset: str,
    query_vector: np.ndarray,
    top_k: int,
    *,
    excluded_item_ids: set[str] | None = None,
) -> list[tuple[str, str]]:
    es = _get_elasticsearch_client()
    es_index = _rag_index_name(dataset)
    if not es.indices.exists(index=es_index):
        raise RuntimeError(
            f"Elasticsearch index '{es_index}' does not exist. Index the dataset first."
        )

    response = es.search(
        index=es_index,
        body={
            "knn": {
                "field": "embedding",
                "query_vector": np.asarray(query_vector, dtype=np.float32).tolist(),
                "k": top_k,
                "num_candidates": max(top_k * 5, 25),
            },
            "_source": ["item_id", "text_repr"],
            "size": max(top_k * 3, 30),
        },
    )

    excluded = excluded_item_ids or set()
    candidates: list[tuple[str, str]] = []
    seen_item_ids: set[str] = set()
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        item_id = source.get("item_id")
        if item_id is None:
            continue
        item_id_str = str(item_id)
        if item_id_str in excluded or item_id_str in seen_item_ids:
            continue
        seen_item_ids.add(item_id_str)
        candidates.append((item_id_str, str(source.get("text_repr", ""))))
        if len(candidates) >= top_k:
            break
    return candidates


def _rag_knn_candidates(
    dataset: str,
    query_text: str,
    top_k: int,
    *,
    excluded_item_ids: set[str] | None = None,
) -> list[tuple[str, str]]:
    embedding_model = _get_sentence_transformer(
        os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    query_vector = embedding_model.encode(query_text, show_progress_bar=False)
    return _rag_knn_candidates_from_vector(
        dataset,
        np.asarray(query_vector, dtype=np.float32),
        top_k,
        excluded_item_ids=excluded_item_ids,
    )


def _rag_seed_item_source(dataset: str, item_id: str) -> dict[str, object]:
    es = _get_elasticsearch_client()
    es_index = _rag_index_name(dataset)
    if not es.indices.exists(index=es_index):
        raise RuntimeError(
            f"Elasticsearch index '{es_index}' does not exist. Index the dataset first."
        )
    response = es.get(index=es_index, id=f"{dataset}::{item_id}")
    source = response.get("_source", {})
    if not isinstance(source, dict):
        raise RuntimeError(f"Elasticsearch source for '{item_id}' is malformed.")
    return source


def _rag_parse_ollama_json(raw_content: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if not match:
        return None
    try:
        import json

        parsed = json.loads(match.group())
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _rag_llm_rerank(
    query_text: str,
    candidates: list[tuple[str, str]],
    top_k: int,
) -> tuple[list[str], str | None]:
    if not candidates:
        return [], None

    try:
        import requests
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("requests is required for the LLM + RAG endpoint.") from exc

    llm_provider = os.getenv("RAG_LLM_PROVIDER", "ollama").strip().lower()
    model_name = (
        os.getenv("RAG_GEMINI_MODEL", "gemini-2.5-flash-lite")
        if llm_provider == "gemini"
        else os.getenv("RAG_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2"))
    )
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    llm_timeout_seconds = float(
        os.getenv(
            "RAG_GEMINI_TIMEOUT_SECONDS" if llm_provider == "gemini" else "RAG_OLLAMA_TIMEOUT_SECONDS",
            "12" if llm_provider == "gemini" else "6",
        )
    )
    rerank_candidate_limit = int(os.getenv("RAG_RERANK_CANDIDATES", "8"))
    rerank_candidates = candidates[: max(top_k, rerank_candidate_limit)]

    prompt = (
        "You are ranking recommendation candidates.\n"
        "Return ONLY valid JSON with this schema:\n"
        '{ "ranked_item_ids": ["id1", "id2"], "explanation": "short explanation" }\n'
        "No extra prose.\n\n"
        f"User request:\n{query_text}\n\n"
        "Candidates:\n"
        + "\n".join(f"- {item_id}: {text_repr}" for item_id, text_repr in rerank_candidates)
    )

    try:
        if llm_provider == "gemini":
            gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini as the RAG reranker.")
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                headers={
                    "x-goog-api-key": gemini_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=llm_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            message_parts = (
                ((((payload.get("candidates") or [{}])[0]).get("content") or {}).get("parts") or [])
            )
            raw_content = "".join(
                str(part.get("text", ""))
                for part in message_parts
                if isinstance(part, dict)
            ).strip()
        else:
            response = requests.post(
                f"{ollama_base_url}/api/chat",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=llm_timeout_seconds,
            )
            response.raise_for_status()
            raw_content = response.json().get("message", {}).get("content", "").strip()
    except Exception:
        fallback_ids = [item_id for item_id, _ in rerank_candidates[:top_k]]
        return (
            fallback_ids,
            f"The {llm_provider} reranker timed out or failed, so the Elasticsearch retrieval order was used directly.",
        )

    parsed = _rag_parse_ollama_json(raw_content)

    if not parsed or not isinstance(parsed.get("ranked_item_ids"), list):
        fallback_ids = [item_id for item_id, _ in rerank_candidates[:top_k]]
        return (
            fallback_ids,
            f"The {llm_provider} response could not be parsed, so the Elasticsearch retrieval order was used directly.",
        )

    ranked_ids = [str(item_id) for item_id in parsed["ranked_item_ids"] if str(item_id).strip()][:top_k]
    if not ranked_ids:
        ranked_ids = [item_id for item_id, _ in rerank_candidates[:top_k]]
    else:
        retrieval_fallback_ids = [item_id for item_id, _ in rerank_candidates]
        seen_ids = set(ranked_ids)
        for item_id in retrieval_fallback_ids:
            if item_id in seen_ids:
                continue
            ranked_ids.append(item_id)
            seen_ids.add(item_id)
            if len(ranked_ids) >= top_k:
                break
        ranked_ids = ranked_ids[:top_k]

    explanation = parsed.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = (
            "The ranking is based on semantic retrieval from Elasticsearch "
            "followed by LLM reranking over the retrieved candidates."
        )

    return ranked_ids, explanation.strip()


# ── Health ─────────────────────────────────────────────────────────────────────
def _model_health_payload(model_slug: str, dataset: str | None = None) -> dict[str, object]:
    model_key = _resolve_model_key(model_slug)
    payload: dict[str, object] = {
        "status": "ok",
        "service": "pfg-recommender",
        "model": model_key,
        "model_slug": model_slug,
        "label": MODEL_KEY_TO_LABEL.get(model_key, model_key),
    }
    if dataset:
        if dataset not in SUPPORTED_DATASETS:
            raise HTTPException(400, f"Unsupported dataset: {dataset}")
        artifact_available = False
        if model_key == "matrix_factorization":
            artifact_available = (MODELS_ROOT / "matrix_factorization" / dataset / "item_factors.npy").exists()
        elif model_key == "two_tower":
            artifact_available = (MODELS_ROOT / "two_tower" / dataset / "item_embeddings.npy").exists()
        elif model_key == "two_tower_wide_deep":
            artifact_available = (
                (MODELS_ROOT / "two_tower_wide_deep" / dataset / "item_embeddings.npy").exists()
                and (MODELS_ROOT / "two_tower_wide_deep" / dataset / "weights.pt").exists()
            )
        elif model_key == "sasrec":
            artifact_available = (
                (MODELS_ROOT / "sasrec" / dataset / "item_embeddings.npy").exists()
                and (MODELS_ROOT / "sasrec" / dataset / "weights.pt").exists()
            )
        elif model_key == "llm_rag":
            es_index = f"pfg_{dataset}_items"
            try:
                artifact_available = bool(_get_elasticsearch_client().indices.exists(index=es_index))
            except Exception:
                artifact_available = False
        payload["dataset"] = dataset
        payload["available"] = artifact_available
    return payload


@app.get(f"{API_V1_PREFIX}/health")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "endpoints": [
            f"{API_V1_PREFIX}/datasets/{{dataset}}/users",
            f"{API_V1_PREFIX}/datasets/{{dataset}}/items/search",
            f"{API_V1_PREFIX}/recommenders/{{model}}/health",
            f"{API_V1_PREFIX}/recommenders/{{model}}/recommendations",
            f"{API_V1_PREFIX}/recommenders/{{model}}/similar-items",
            "/search",
            "/mf",
            "/mf/similar",
            "/two_tower",
            "/two_tower/similar",
            "/two_tower_wide_deep",
            "/two_tower_wide_deep/similar",
            "/sasrec",
            "/sasrec/similar",
            "/rag",
            "/rag/similar",
        ],
    }


@app.get(f"{API_V1_PREFIX}/datasets/{{dataset}}/users", response_model=DatasetUsersResponse)
@app.get("/dataset-users", response_model=DatasetUsersResponse)
def dataset_users(dataset: str, limit: int = 25):
    if dataset not in SUPPORTED_DATASETS:
        raise HTTPException(400, f"Unsupported dataset: {dataset}")

    options, total_available = _dataset_user_options(dataset)
    trimmed = options[: max(1, min(limit, 200))]
    return DatasetUsersResponse(
        dataset=dataset,
        users=trimmed,
        total_available=total_available,
    )


@app.get(f"{API_V1_PREFIX}/health/detailed")
@app.get("/health/detailed")
def health_detailed():
    datasets: dict[str, dict[str, bool]] = {}
    for dataset in sorted(SUPPORTED_DATASETS):
        dataset_dir = DATA_ROOT / dataset
        datasets[dataset] = {
            "items_parquet": (dataset_dir / "items.parquet").exists(),
            "interactions_parquet": (dataset_dir / "interactions.parquet").exists(),
        }

    return {
        "status": "ok",
        "paths": {
            "models_root": str(MODELS_ROOT),
            "data_root": str(DATA_ROOT),
        },
        "implemented_endpoints": {
            f"{API_V1_PREFIX}/datasets/{{dataset}}/users": True,
            f"{API_V1_PREFIX}/datasets/{{dataset}}/items/search": True,
            f"{API_V1_PREFIX}/recommenders/{{model}}/health": True,
            f"{API_V1_PREFIX}/recommenders/{{model}}/recommendations": True,
            f"{API_V1_PREFIX}/recommenders/{{model}}/similar-items": True,
            "/search": True,
            "/mf": True,
            "/mf/similar": True,
            "/two_tower": True,
            "/two_tower/similar": True,
            "/two_tower_wide_deep": True,
            "/two_tower_wide_deep/similar": True,
            "/sasrec": True,
            "/sasrec/similar": True,
            "/rag": True,
            "/rag/similar": True,
        },
        "datasets": datasets,
    }


@app.get(f"{API_V1_PREFIX}/recommenders/{{model_slug}}/health")
def recommender_model_health(model_slug: str, dataset: str | None = None):
    return _model_health_payload(model_slug, dataset)


# ── /search — Fuzzy text search in items.parquet ───────────────────────────────
@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search_items(req: SearchRequest):
    """Entity lookup backed by Elasticsearch, with a local Parquet fallback for robustness."""
    with _timed_span(req.trace_id, "recommender.search", dataset=req.dataset, top_k=req.top_k):
        if req.dataset not in SUPPORTED_DATASETS:
            raise HTTPException(400, f"Unsupported dataset: {req.dataset}")
        search_backend = "elasticsearch"
        try:
            results = _search_items_from_elasticsearch(req)
        except Exception as exc:
            search_backend = "parquet_fallback"
            log.warning(
                "[SEARCH] trace_id=%r dataset=%r elasticsearch lookup failed, falling back to Parquet: %s",
                req.trace_id,
                req.dataset,
                exc,
            )
            results = _search_items_from_parquet(req)
        else:
            if not results:
                search_backend = "parquet_fallback"
                results = _search_items_from_parquet(req)

        log.info(
            "[SEARCH] trace_id=%r query=%r dataset=%r backend=%r n_results=%s preview_titles=%s",
            req.trace_id,
            req.query,
            req.dataset,
            search_backend,
            len(results),
            [r.title for r in results[:5]],
        )
        return SearchResponse(query=req.query, dataset=req.dataset, results=results, trace_id=req.trace_id)


@app.get(f"{API_V1_PREFIX}/datasets/{{dataset}}/items/search", response_model=SearchResponse, tags=["Search"])
def search_items_v1(dataset: str, q: str, limit: int = 10, trace_id: str | None = None):
    request = SearchRequest(query=q, dataset=dataset, top_k=limit, trace_id=trace_id)
    return search_items(request)


def _dispatch_recommendation(model_slug: str, req: RecommendRequest):
    model_key = _resolve_model_key(model_slug)
    if model_key == "matrix_factorization":
        return mf_recommend(req)
    if model_key == "two_tower":
        return two_tower_recommend(req)
    if model_key == "two_tower_wide_deep":
        return two_tower_wide_deep_recommend(req)
    if model_key == "sasrec":
        return sasrec_recommend(req)
    if model_key == "llm_rag":
        return rag_recommend(req)
    raise HTTPException(404, f"Unknown recommender model '{model_slug}'.")


def _dispatch_similarity(model_slug: str, req: SimilarRequest):
    model_key = _resolve_model_key(model_slug)
    if model_key == "matrix_factorization":
        return mf_similar(req)
    if model_key == "two_tower":
        return two_tower_similar(req)
    if model_key == "two_tower_wide_deep":
        return two_tower_wide_deep_similar(req)
    if model_key == "sasrec":
        return sasrec_similar(req)
    if model_key == "llm_rag":
        return rag_similar(req)
    raise HTTPException(404, f"Unknown recommender model '{model_slug}'.")


@app.post(
    f"{API_V1_PREFIX}/recommenders/{{model_slug}}/recommendations",
    response_model=RecommendResponse,
    tags=["Versioned Recommenders"],
)
def recommend_v1(model_slug: str, req: RecommendRequest):
    return _dispatch_recommendation(model_slug, req)


@app.post(
    f"{API_V1_PREFIX}/recommenders/{{model_slug}}/similar-items",
    response_model=SimilarResponse,
    tags=["Versioned Recommenders"],
)
def similar_items_v1(model_slug: str, req: SimilarRequest):
    return _dispatch_similarity(model_slug, req)


# ── /mf — Matrix Factorization ─────────────────────────────────────────────────
mf_router = APIRouter(prefix="/mf", tags=["Matrix Factorization"])

@mf_router.post("", response_model=RecommendResponse, summary="MF recommendations")
def mf_recommend(req: RecommendRequest):
    """Matrix Factorization (ALS) recommendations using cosine similarity on user/item factors."""
    with _timed_span(
        req.trace_id,
        "recommender.mf_recommend",
        dataset=req.dataset,
        user_id=req.user_id,
        top_k=req.top_k,
    ):
        if req.dataset not in SUPPORTED_DATASETS:
            raise HTTPException(400, f"Unsupported dataset: {req.dataset}")

        model_dir = MODELS_ROOT / "matrix_factorization" / req.dataset
        user_factors_path = model_dir / "user_factors.npy"
        item_factors_path = model_dir / "item_factors.npy"

        if not item_factors_path.exists():
            raise HTTPException(404, f"MF weights not found for dataset '{req.dataset}' at {item_factors_path}")

        _ = _load_npy(str(item_factors_path))
        cold_start = True
        indices: list[int]
        explanation: str | None = None

        candidate_n = max(req.top_k * 20, 200)
        prompt_include_genres, prompt_exclude_genres = _genre_constraints_from_prompt(req.prompt)

        if user_factors_path.exists():
            user_factors = _load_npy(str(user_factors_path))
            dataset_user_id = (req.dataset_user_id or "").strip()
            dataset_user_map = _dataset_user_index_map(req.dataset)

            if dataset_user_id and dataset_user_id in dataset_user_map:
                user_idx = dataset_user_map[dataset_user_id]
                if user_idx < len(user_factors):
                    user_vec = user_factors[user_idx]
                    item_factors = _load_npy(str(item_factors_path))
                    scores = item_factors @ user_vec
                    seen = set(_dataset_seen_item_indices(req.dataset, dataset_user_id))
                    if seen:
                        valid_seen = _clip_valid_indices(seen, len(scores))
                        dropped_seen = len(seen) - len(valid_seen)
                        if dropped_seen > 0:
                            log.warning(
                                "[MF] trace_id=%r dataset=%r dataset_user_id=%r dropped %s out-of-bounds seen-item indices",
                                req.trace_id,
                                req.dataset,
                                dataset_user_id,
                                dropped_seen,
                            )
                        if valid_seen:
                            scores[valid_seen] = -1e9
                    top_n = max(req.top_k * 20, 200)
                    indices = np.argsort(-scores)[:top_n].tolist()
                    genre_filtered = _filter_candidates_by_genres(
                        req.dataset,
                        indices,
                        prompt_include_genres,
                        prompt_exclude_genres,
                    )
                    if len(genre_filtered) >= req.top_k:
                        indices = genre_filtered
                    indices = _rerank_by_prompt(req.dataset, indices, req.prompt, req.top_k)
                    cold_start = False
                    explanation = (
                        f"The ranking uses the offline-trained Matrix Factorization user profile "
                        f"for dataset user `{dataset_user_id}` and then applies prompt-aware reranking."
                    )
                    log.info(
                        "[MF] trace_id=%r dataset_user_id=%r dataset=%r using offline-trained user factors",
                        req.trace_id,
                        dataset_user_id,
                        req.dataset,
                    )
                else:
                    log.warning(
                        "[MF] trace_id=%r dataset_user_id=%r resolved user_idx=%s outside user_factors bounds",
                        req.trace_id,
                        dataset_user_id,
                        user_idx,
                    )
                    indices = _popularity_fallback(req.dataset, candidate_n)
                    genre_filtered = _filter_candidates_by_genres(
                        req.dataset,
                        indices,
                        prompt_include_genres,
                        prompt_exclude_genres,
                    )
                    if len(genre_filtered) >= req.top_k:
                        indices = genre_filtered
                    indices = _rerank_by_prompt(req.dataset, indices, req.prompt, req.top_k)
                    cold_start = True
            else:
                log.info(
                    "[MF] trace_id=%r user_id=%r dataset=%r external user, using popularity fallback",
                    req.trace_id,
                    req.user_id,
                    req.dataset,
                )
                indices = _popularity_fallback(req.dataset, candidate_n)
                genre_filtered = _filter_candidates_by_genres(
                    req.dataset,
                    indices,
                    prompt_include_genres,
                    prompt_exclude_genres,
                )
                if len(genre_filtered) >= req.top_k:
                    indices = genre_filtered
                indices = _rerank_by_prompt(req.dataset, indices, req.prompt, req.top_k)
                cold_start = True
                explanation = (
                    "The ranking is in cold-start mode because no existing dataset user profile was selected."
                )
        else:
            log.warning(
                "[MF] trace_id=%r no user_factors found for dataset=%r, using popularity fallback",
                req.trace_id,
                req.dataset,
            )
            indices = _popularity_fallback(req.dataset, candidate_n)
            genre_filtered = _filter_candidates_by_genres(
                req.dataset,
                indices,
                prompt_include_genres,
                prompt_exclude_genres,
            )
            if len(genre_filtered) >= req.top_k:
                indices = genre_filtered
            indices = _rerank_by_prompt(req.dataset, indices, req.prompt, req.top_k)
            explanation = (
                "The ranking is in cold-start mode because user factors are not available for this dataset."
            )

        response = _build_response(
            req.user_id,
            req.dataset,
            "matrix_factorization",
            indices,
            cold_start,
            trace_id=req.trace_id,
        )
        response.explanation = explanation
        log.info(
            "[MF] trace_id=%r prompt=%r tokens=%s genre_targets=%s",
            req.trace_id,
            req.prompt,
            _prompt_tokens(req.prompt),
            {"include": sorted(prompt_include_genres), "exclude": sorted(prompt_exclude_genres)},
        )
        _log_recommendation_preview(
            "[MF]",
            req.dataset,
            "matrix_factorization",
            response.items,
            cold_start,
            trace_id=req.trace_id,
        )
        return response

app.include_router(mf_router)


# ── /mf/similar — MF item-to-item similarity ───────────────────────────────────
@mf_router.post("/similar", response_model=SimilarResponse, summary="MF item similarity")
def mf_similar(req: SimilarRequest):
    """Find items most similar to a given item using MF item factor cosine similarity."""
    with _timed_span(req.trace_id, "recommender.mf_similar", dataset=req.dataset, top_k=req.top_k):
        if req.dataset not in SUPPORTED_DATASETS:
            raise HTTPException(400, f"Unsupported dataset: {req.dataset}")

        model_dir = MODELS_ROOT / "matrix_factorization" / req.dataset
        item_factors_path = model_dir / "item_factors.npy"
        if not item_factors_path.exists():
            raise HTTPException(404, f"MF weights not found for dataset '{req.dataset}'")

        items_df = _load_items(req.dataset)
        title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
        genre_col = _find_optional_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])

        mask = items_df[title_col].astype(str).str.contains(req.item_title, case=False, na=False, regex=False)
        matches = items_df[mask]
        if matches.empty:
            raise HTTPException(404, f"Item '{req.item_title}' not found in dataset '{req.dataset}'")

        seed_idx = int(matches.index[0])
        seed_title = str(matches.iloc[0][title_col])
        log.info(
            "[MF/SIMILAR] trace_id=%r seed=%r idx=%s dataset=%r",
            req.trace_id,
            seed_title,
            seed_idx,
            req.dataset,
        )

        item_factors = _load_npy(str(item_factors_path))
        seed_vec = item_factors[seed_idx]
        top_indices = _cosine_topk(seed_vec, item_factors, req.top_k + 1)
        top_indices = [i for i in top_indices if i != seed_idx][: req.top_k]

        similar_items = []
        for rank, idx in enumerate(top_indices):
            if idx >= len(items_df):
                continue
            row = items_df.iloc[idx]
            similar_items.append(RecommendedItem(
                title=str(row[title_col]),
                score=round(1.0 - rank / req.top_k, 3),
                genres=str(row.get(genre_col, "")).strip() if genre_col else "",
            ))

        response = SimilarResponse(
            seed_title=seed_title,
            dataset=req.dataset,
            model="matrix_factorization",
            items=similar_items,
            trace_id=req.trace_id,
        )
        _log_recommendation_preview(
            "[MF/SIMILAR]",
            req.dataset,
            "matrix_factorization",
            response.items,
            trace_id=req.trace_id,
        )
        return response

# ── /two_tower — placeholder ───────────────────────────────────────────────────
@app.post("/two_tower", response_model=RecommendResponse, tags=["Two-Tower"])
def two_tower_recommend(req: RecommendRequest):
    with _timed_span(
        req.trace_id,
        "recommender.two_tower_recommend",
        dataset=req.dataset,
        user_id=req.user_id,
        top_k=req.top_k,
    ):
        if req.dataset not in SUPPORTED_DATASETS:
            raise HTTPException(400, f"Unsupported dataset: {req.dataset}")

        model_dir = MODELS_ROOT / "two_tower" / req.dataset
        item_embeddings_path = model_dir / "item_embeddings.npy"
        if not item_embeddings_path.exists():
            raise HTTPException(
                404,
                f"Two-Tower item embeddings not found for dataset '{req.dataset}' at {item_embeddings_path}",
            )

        item_embeddings = _load_npy(str(item_embeddings_path))
        dataset_user_id = (req.dataset_user_id or "").strip()
        dataset_user_map = _dataset_user_index_map(req.dataset)
        candidate_n = max(req.top_k * 20, 200)
        prompt_include_genres, prompt_exclude_genres = _genre_constraints_from_prompt(req.prompt)
        cold_start = True
        explanation: str | None = None

        if dataset_user_id and dataset_user_id in dataset_user_map:
            user_idx = dataset_user_map[dataset_user_id]
            try:
                user_embeddings = _two_tower_user_embeddings(req.dataset)
            except Exception as exc:
                log.warning(
                    "[TWO_TOWER] trace_id=%r failed to load offline user embeddings for dataset=%r: %s",
                    req.trace_id,
                    req.dataset,
                    exc,
                )
                user_history_indices = _dataset_user_sequence_indices(req.dataset, dataset_user_id)
                derived_user_vec = _history_profile_vector(
                    item_embeddings,
                    user_history_indices,
                    sequential=False,
                )
                if derived_user_vec is None:
                    candidate_item_ids = _popular_item_ids(req.dataset, candidate_n)
                    genre_filtered = _filter_item_ids_by_genres(
                        req.dataset,
                        candidate_item_ids,
                        prompt_include_genres,
                        prompt_exclude_genres,
                    )
                    if len(genre_filtered) >= req.top_k:
                        candidate_item_ids = genre_filtered
                    candidate_item_ids = _rerank_item_ids_by_prompt(req.dataset, candidate_item_ids, req.prompt, req.top_k)
                    explanation = (
                        "The ranking fell back to cold-start mode because the offline-trained Two-Tower "
                        "user embeddings were not available and no usable dataset-user history could be projected."
                    )
                else:
                    scores = item_embeddings @ derived_user_vec
                    seen = set(_dataset_seen_item_indices(req.dataset, dataset_user_id))
                    if seen:
                        valid_seen = _clip_valid_indices(seen, len(scores))
                        if valid_seen:
                            scores[valid_seen] = -1e9
                    top_indices = np.argsort(-scores)[:candidate_n].tolist()
                    candidate_item_ids = _indices_to_item_ids(req.dataset, top_indices)
                    genre_filtered = _filter_item_ids_by_genres(
                        req.dataset,
                        candidate_item_ids,
                        prompt_include_genres,
                        prompt_exclude_genres,
                    )
                    if len(genre_filtered) >= req.top_k:
                        candidate_item_ids = genre_filtered
                    candidate_item_ids = _rerank_item_ids_by_prompt(
                        req.dataset,
                        candidate_item_ids,
                        req.prompt,
                        req.top_k,
                    )
                    cold_start = False
                    explanation = (
                        f"The ranking uses a history-derived dataset-user representation for `{dataset_user_id}` "
                        "projected into the offline-trained Two-Tower item embedding space, followed by prompt-aware reranking."
                    )
            else:
                if user_idx < len(user_embeddings):
                    user_vec = user_embeddings[user_idx]
                    scores = item_embeddings @ user_vec
                    seen = set(_dataset_seen_item_indices(req.dataset, dataset_user_id))
                    if seen:
                        valid_seen = _clip_valid_indices(seen, len(scores))
                        if valid_seen:
                            scores[valid_seen] = -1e9
                    top_indices = np.argsort(-scores)[:candidate_n].tolist()
                    candidate_item_ids = _indices_to_item_ids(req.dataset, top_indices)
                    genre_filtered = _filter_item_ids_by_genres(
                        req.dataset,
                        candidate_item_ids,
                        prompt_include_genres,
                        prompt_exclude_genres,
                    )
                    if len(genre_filtered) >= req.top_k:
                        candidate_item_ids = genre_filtered
                    candidate_item_ids = _rerank_item_ids_by_prompt(
                        req.dataset,
                        candidate_item_ids,
                        req.prompt,
                        req.top_k,
                    )
                    cold_start = False
                    explanation = (
                        f"The ranking uses the offline-trained Two-Tower user profile "
                        f"for dataset user `{dataset_user_id}` and then applies prompt-aware reranking."
                    )
                else:
                    candidate_item_ids = _popular_item_ids(req.dataset, candidate_n)
                    genre_filtered = _filter_item_ids_by_genres(req.dataset, candidate_item_ids, prompt_genres)
                    if len(genre_filtered) >= req.top_k:
                        candidate_item_ids = genre_filtered
                    candidate_item_ids = _rerank_item_ids_by_prompt(req.dataset, candidate_item_ids, req.prompt, req.top_k)
                    explanation = (
                        "The ranking is in cold-start mode because the selected dataset user was outside the "
                        "stored Two-Tower user embedding matrix."
                    )
        else:
            candidate_item_ids = _popular_item_ids(req.dataset, candidate_n)
            genre_filtered = _filter_item_ids_by_genres(
                req.dataset,
                candidate_item_ids,
                prompt_include_genres,
                prompt_exclude_genres,
            )
            if len(genre_filtered) >= req.top_k:
                candidate_item_ids = genre_filtered
            candidate_item_ids = _rerank_item_ids_by_prompt(req.dataset, candidate_item_ids, req.prompt, req.top_k)
            explanation = (
                "The ranking is in cold-start mode because no existing dataset user profile was selected."
            )

        response = _build_response_from_item_ids(
            req.user_id,
            req.dataset,
            "two_tower",
            candidate_item_ids,
            cold_start,
            trace_id=req.trace_id,
            explanation=explanation,
        )
        log.info(
            "[TWO_TOWER] trace_id=%r prompt=%r tokens=%s genre_targets=%s",
            req.trace_id,
            req.prompt,
            _prompt_tokens(req.prompt),
            {"include": sorted(prompt_include_genres), "exclude": sorted(prompt_exclude_genres)},
        )
        _log_recommendation_preview(
            "[TWO_TOWER]",
            req.dataset,
            "two_tower",
            response.items,
            cold_start,
            trace_id=req.trace_id,
        )
        return response


@app.post("/two_tower/similar", response_model=SimilarResponse, tags=["Two-Tower"])
def two_tower_similar(req: SimilarRequest):
    with _timed_span(req.trace_id, "recommender.two_tower_similar", dataset=req.dataset, top_k=req.top_k):
        if req.dataset not in SUPPORTED_DATASETS:
            raise HTTPException(400, f"Unsupported dataset: {req.dataset}")

        model_dir = MODELS_ROOT / "two_tower" / req.dataset
        item_embeddings_path = model_dir / "item_embeddings.npy"
        if not item_embeddings_path.exists():
            raise HTTPException(
                404,
                f"Two-Tower item embeddings not found for dataset '{req.dataset}' at {item_embeddings_path}",
            )

        items_df = _load_items(req.dataset)
        item_id_col = _get_item_id_col(items_df)
        title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
        mask = items_df[title_col].astype(str).str.contains(req.item_title, case=False, na=False, regex=False)
        matches = items_df[mask]
        if matches.empty:
            raise HTTPException(404, f"Item '{req.item_title}' not found in dataset '{req.dataset}'")

        seed_row = matches.iloc[0]
        seed_title = str(seed_row[title_col])
        seed_item_id = str(seed_row[item_id_col])
        item_map = _dataset_item_index_map(req.dataset)
        if seed_item_id not in item_map:
            raise HTTPException(404, f"Item '{seed_title}' is not represented in the Two-Tower interaction space.")

        seed_idx = item_map[seed_item_id]
        item_embeddings = _load_npy(str(item_embeddings_path))
        if seed_idx >= len(item_embeddings):
            raise HTTPException(500, f"Seed item index {seed_idx} is outside Two-Tower embedding bounds.")

        seed_vec = item_embeddings[seed_idx]
        top_indices = _cosine_topk(seed_vec, item_embeddings, req.top_k + 1)
        top_indices = [idx for idx in top_indices if idx != seed_idx][: req.top_k]
        item_ids = _indices_to_item_ids(req.dataset, top_indices)
        similar_items = _similar_items_from_item_ids(req.dataset, item_ids, req.top_k)

        response = SimilarResponse(
            seed_title=seed_title,
            dataset=req.dataset,
            model="two_tower",
            items=similar_items,
            trace_id=req.trace_id,
        )
        _log_recommendation_preview(
            "[TWO_TOWER/SIMILAR]",
            req.dataset,
            "two_tower",
            response.items,
            trace_id=req.trace_id,
        )
        return response


@app.post("/two_tower_wide_deep", tags=["Two-Tower + Wide&Deep"])
def two_tower_wide_deep_recommend(req: RecommendRequest):
    with _timed_span(
        req.trace_id,
        "recommender.two_tower_wide_deep_recommend",
        dataset=req.dataset,
        user_id=req.user_id,
        top_k=req.top_k,
    ):
        if req.dataset not in SUPPORTED_DATASETS:
            raise HTTPException(400, f"Unsupported dataset: {req.dataset}")

        model_dir = MODELS_ROOT / "two_tower_wide_deep" / req.dataset
        item_embeddings_path = model_dir / "item_embeddings.npy"
        weights_path = model_dir / "weights.pt"
        if not item_embeddings_path.exists() or not weights_path.exists():
            raise HTTPException(
                404,
                f"Run not trained: Two-Tower + Wide&Deep is not available for dataset '{req.dataset}'.",
            )

        item_embeddings = _load_npy(str(item_embeddings_path))
        dataset_user_id = (req.dataset_user_id or "").strip()
        dataset_user_map = _dataset_user_index_map(req.dataset)
        candidate_n = max(req.top_k * 20, 200)
        prompt_include_genres, prompt_exclude_genres = _genre_constraints_from_prompt(req.prompt)
        cold_start = True
        explanation: str | None = None

        if dataset_user_id and dataset_user_id in dataset_user_map:
            user_idx = dataset_user_map[dataset_user_id]
            user_history_indices = _dataset_user_sequence_indices(req.dataset, dataset_user_id)
            profile_vector = _history_profile_vector(item_embeddings, user_history_indices, sequential=False)
            try:
                arrays = _two_tower_wide_deep_arrays(req.dataset)
                user_matrix = arrays["user_emb.weight"]
                exact_user_vec = user_matrix[user_idx] if user_idx < len(user_matrix) else None
            except Exception as exc:
                log.warning(
                    "[TTWD] trace_id=%r failed to load full ranking head for dataset=%r: %s",
                    req.trace_id,
                    req.dataset,
                    exc,
                )
                exact_user_vec = None

            user_vec = exact_user_vec if exact_user_vec is not None else profile_vector
            if user_vec is not None:
                scores = item_embeddings @ user_vec
                seen = set(_dataset_seen_item_indices(req.dataset, dataset_user_id))
                if seen:
                    valid_seen = _clip_valid_indices(seen, len(scores))
                    if valid_seen:
                        scores[valid_seen] = -1e9
                top_indices = np.argsort(-scores)[:candidate_n].tolist()
                if exact_user_vec is not None:
                    try:
                        top_indices = _score_two_tower_wide_deep_candidates(
                            req.dataset,
                            user_vec,
                            user_idx,
                            top_indices,
                        )
                    except Exception as exc:
                        log.warning(
                            "[TTWD] trace_id=%r failed to apply Wide&Deep reranking head for dataset=%r: %s",
                            req.trace_id,
                            req.dataset,
                            exc,
                        )
                candidate_item_ids = _indices_to_item_ids(req.dataset, top_indices)
                genre_filtered = _filter_item_ids_by_genres(
                    req.dataset,
                    candidate_item_ids,
                    prompt_include_genres,
                    prompt_exclude_genres,
                )
                if len(genre_filtered) >= req.top_k:
                    candidate_item_ids = genre_filtered
                candidate_item_ids = _rerank_item_ids_by_prompt(req.dataset, candidate_item_ids, req.prompt, req.top_k)
                cold_start = False
                if exact_user_vec is not None:
                    explanation = (
                        f"The ranking uses the offline-trained Two-Tower + Wide&Deep user profile "
                        f"for dataset user `{dataset_user_id}`, including candidate retrieval and ranking-head reranking."
                    )
                else:
                    explanation = (
                        f"The ranking uses a history-derived dataset-user representation for `{dataset_user_id}` "
                        "projected into the offline-trained Two-Tower + Wide&Deep item embedding space."
                    )
            else:
                candidate_item_ids = _popular_item_ids(req.dataset, candidate_n)
                genre_filtered = _filter_item_ids_by_genres(
                    req.dataset,
                    candidate_item_ids,
                    prompt_include_genres,
                    prompt_exclude_genres,
                )
                if len(genre_filtered) >= req.top_k:
                    candidate_item_ids = genre_filtered
                candidate_item_ids = _rerank_item_ids_by_prompt(req.dataset, candidate_item_ids, req.prompt, req.top_k)
                explanation = (
                    "The ranking is in cold-start mode because no usable dataset-user profile could be built."
                )
        else:
            candidate_item_ids = _popular_item_ids(req.dataset, candidate_n)
            genre_filtered = _filter_item_ids_by_genres(
                req.dataset,
                candidate_item_ids,
                prompt_include_genres,
                prompt_exclude_genres,
            )
            if len(genre_filtered) >= req.top_k:
                candidate_item_ids = genre_filtered
            candidate_item_ids = _rerank_item_ids_by_prompt(req.dataset, candidate_item_ids, req.prompt, req.top_k)
            explanation = (
                "The ranking is in cold-start mode because no existing dataset user profile was selected."
            )

        response = _build_response_from_item_ids(
            req.user_id,
            req.dataset,
            "two_tower_wide_deep",
            candidate_item_ids,
            cold_start,
            trace_id=req.trace_id,
            explanation=explanation,
        )
        _log_recommendation_preview(
            "[TTWD]",
            req.dataset,
            "two_tower_wide_deep",
            response.items,
            cold_start,
            trace_id=req.trace_id,
        )
        return response


@app.post("/two_tower_wide_deep/similar", response_model=SimilarResponse, tags=["Two-Tower + Wide&Deep"])
def two_tower_wide_deep_similar(req: SimilarRequest):
    model_dir = MODELS_ROOT / "two_tower_wide_deep" / req.dataset
    item_embeddings_path = model_dir / "item_embeddings.npy"
    if not item_embeddings_path.exists():
        raise HTTPException(
            404,
            f"Run not trained: Two-Tower + Wide&Deep is not available for dataset '{req.dataset}'.",
        )
    forwarded = SimilarRequest(
        item_title=req.item_title,
        dataset=req.dataset,
        top_k=req.top_k,
        trace_id=req.trace_id,
    )
    # Reuse the same retrieval geometry as Two-Tower because the stage-1 item embeddings are shared.
    return two_tower_similar(forwarded).model_copy(update={"model": "two_tower_wide_deep"})


# ── /sasrec — placeholder ──────────────────────────────────────────────────────
@app.post("/sasrec", tags=["SASRec"])
def sasrec_recommend(req: RecommendRequest):
    with _timed_span(
        req.trace_id,
        "recommender.sasrec_recommend",
        dataset=req.dataset,
        user_id=req.user_id,
        top_k=req.top_k,
    ):
        if req.dataset not in SUPPORTED_DATASETS:
            raise HTTPException(400, f"Unsupported dataset: {req.dataset}")

        model_dir = MODELS_ROOT / "sasrec" / req.dataset
        item_embeddings_path = model_dir / "item_embeddings.npy"
        weights_path = model_dir / "weights.pt"
        if not item_embeddings_path.exists() or not weights_path.exists():
            raise HTTPException(
                404,
                f"Run not trained: SASRec is not available for dataset '{req.dataset}'.",
            )

        item_embeddings = _load_npy(str(item_embeddings_path))
        dataset_user_id = (req.dataset_user_id or "").strip()
        dataset_user_map = _dataset_user_index_map(req.dataset)
        candidate_n = max(req.top_k * 20, 200)
        prompt_include_genres, prompt_exclude_genres = _genre_constraints_from_prompt(req.prompt)
        cold_start = True
        explanation: str | None = None

        if dataset_user_id and dataset_user_id in dataset_user_map:
            sequence = list(_dataset_user_sequence_indices(req.dataset, dataset_user_id))
            user_vec: np.ndarray | None = None
            try:
                serving_model = _get_sasrec_serving_model(req.dataset)
                sequence_1_indexed = [idx + 1 for idx in sequence]
                user_vec = serving_model.encode_sequence(sequence_1_indexed)
                explanation = (
                    f"The ranking uses the offline-trained SASRec sequence encoder for dataset user `{dataset_user_id}`."
                )
            except Exception as exc:
                log.warning(
                    "[SASREC] trace_id=%r failed to load sequence model for dataset=%r: %s",
                    req.trace_id,
                    req.dataset,
                    exc,
                )
                user_vec = _history_profile_vector(item_embeddings, sequence, sequential=True)
                if user_vec is not None:
                    explanation = (
                        f"The ranking uses a recency-weighted sequence profile for dataset user `{dataset_user_id}` "
                        "projected into the offline-trained SASRec item embedding space."
                    )

            if user_vec is not None:
                scores = item_embeddings @ user_vec
                seen = set(_dataset_seen_item_indices(req.dataset, dataset_user_id))
                if seen:
                    valid_seen = _clip_valid_indices(seen, len(scores))
                    if valid_seen:
                        scores[valid_seen] = -1e9
                top_indices = np.argsort(-scores)[:candidate_n].tolist()
                candidate_item_ids = _indices_to_item_ids(req.dataset, top_indices)
                genre_filtered = _filter_item_ids_by_genres(
                    req.dataset,
                    candidate_item_ids,
                    prompt_include_genres,
                    prompt_exclude_genres,
                )
                if len(genre_filtered) >= req.top_k:
                    candidate_item_ids = genre_filtered
                candidate_item_ids = _rerank_item_ids_by_prompt(req.dataset, candidate_item_ids, req.prompt, req.top_k)
                cold_start = False
            else:
                candidate_item_ids = _popular_item_ids(req.dataset, candidate_n)
                genre_filtered = _filter_item_ids_by_genres(
                    req.dataset,
                    candidate_item_ids,
                    prompt_include_genres,
                    prompt_exclude_genres,
                )
                if len(genre_filtered) >= req.top_k:
                    candidate_item_ids = genre_filtered
                candidate_item_ids = _rerank_item_ids_by_prompt(req.dataset, candidate_item_ids, req.prompt, req.top_k)
                explanation = (
                    "The ranking is in cold-start mode because no usable sequential dataset-user profile could be built."
                )
        else:
            candidate_item_ids = _popular_item_ids(req.dataset, candidate_n)
            genre_filtered = _filter_item_ids_by_genres(
                req.dataset,
                candidate_item_ids,
                prompt_include_genres,
                prompt_exclude_genres,
            )
            if len(genre_filtered) >= req.top_k:
                candidate_item_ids = genre_filtered
            candidate_item_ids = _rerank_item_ids_by_prompt(req.dataset, candidate_item_ids, req.prompt, req.top_k)
            explanation = (
                "The ranking is in cold-start mode because no existing dataset user profile was selected."
            )

        response = _build_response_from_item_ids(
            req.user_id,
            req.dataset,
            "sasrec",
            candidate_item_ids,
            cold_start,
            trace_id=req.trace_id,
            explanation=explanation,
        )
        _log_recommendation_preview(
            "[SASREC]",
            req.dataset,
            "sasrec",
            response.items,
            cold_start,
            trace_id=req.trace_id,
        )
        return response


@app.post("/sasrec/similar", response_model=SimilarResponse, tags=["SASRec"])
def sasrec_similar(req: SimilarRequest):
    model_dir = MODELS_ROOT / "sasrec" / req.dataset
    item_embeddings_path = model_dir / "item_embeddings.npy"
    if not item_embeddings_path.exists():
        raise HTTPException(
            404,
            f"Run not trained: SASRec is not available for dataset '{req.dataset}'.",
        )

    items_df = _load_items(req.dataset)
    item_id_col = _get_item_id_col(items_df)
    title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
    mask = items_df[title_col].astype(str).str.contains(req.item_title, case=False, na=False, regex=False)
    matches = items_df[mask]
    if matches.empty:
        raise HTTPException(404, f"Item '{req.item_title}' not found in dataset '{req.dataset}'")

    seed_row = matches.iloc[0]
    seed_title = str(seed_row[title_col])
    seed_item_id = str(seed_row[item_id_col])
    item_map = _dataset_item_index_map(req.dataset)
    if seed_item_id not in item_map:
        raise HTTPException(404, f"Item '{seed_title}' is not represented in the SASRec interaction space.")

    seed_idx = item_map[seed_item_id]
    item_embeddings = _load_npy(str(item_embeddings_path))
    if seed_idx >= len(item_embeddings):
        raise HTTPException(500, f"Seed item index {seed_idx} is outside SASRec embedding bounds.")

    seed_vec = item_embeddings[seed_idx]
    top_indices = _cosine_topk(seed_vec, item_embeddings, req.top_k + 1)
    top_indices = [idx for idx in top_indices if idx != seed_idx][: req.top_k]
    item_ids = _indices_to_item_ids(req.dataset, top_indices)
    similar_items = _similar_items_from_item_ids(req.dataset, item_ids, req.top_k)
    return SimilarResponse(
        seed_title=seed_title,
        dataset=req.dataset,
        model="sasrec",
        items=similar_items,
        trace_id=req.trace_id,
    )


@app.post("/rag", response_model=RecommendResponse, tags=["LLM + RAG"])
def rag_recommend(req: RecommendRequest):
    with _timed_span(
        req.trace_id,
        "recommender.rag_recommend",
        dataset=req.dataset,
        user_id=req.user_id,
        top_k=req.top_k,
    ):
        if req.dataset not in SUPPORTED_DATASETS:
            raise HTTPException(400, f"Unsupported dataset: {req.dataset}")

        try:
            query_text = _rag_build_query_text(req)
            dataset_user_id = (req.dataset_user_id or "").strip()
            profile_context = (
                _dataset_user_profile_context(req.dataset, dataset_user_id)
                if dataset_user_id
                else ""
            )
            excluded_item_ids = (
                set(_dataset_seen_item_ids(req.dataset, dataset_user_id))
                if dataset_user_id
                else set()
            )
            candidate_count = max(req.top_k * 2, 18)
            candidates = _rag_knn_candidates(
                req.dataset,
                query_text,
                candidate_count,
                excluded_item_ids=excluded_item_ids,
            )
            if not candidates:
                raise HTTPException(
                    404,
                    f"No semantic candidates were found for dataset '{req.dataset}'.",
                )

            ranked_item_ids, rerank_explanation = _rag_llm_rerank(
                query_text,
                candidates,
                req.top_k,
            )
            cold_start = not bool(profile_context)
            explanation_parts = []
            if cold_start:
                explanation_parts.append(
                    "The ranking uses semantic retrieval and LLM reranking driven only by the current prompt."
                )
            else:
                explanation_parts.append(
                    f"The ranking uses semantic retrieval conditioned on dataset user `{dataset_user_id}` and excludes already seen items before LLM reranking."
                )
            if rerank_explanation:
                explanation_parts.append(f"LLM reranker summary: {rerank_explanation}")
            explanation = " ".join(explanation_parts)
            response = _build_response_from_item_ids(
                req.user_id,
                req.dataset,
                "llm_rag",
                ranked_item_ids,
                cold_start=cold_start,
                trace_id=req.trace_id,
                explanation=explanation,
            )
            _log_recommendation_preview(
                "[RAG]",
                req.dataset,
                "llm_rag",
                response.items,
                cold_start=cold_start,
                trace_id=req.trace_id,
            )
            return response
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"LLM + RAG recommendation error: {exc}") from exc


@app.post("/rag/similar", response_model=SimilarResponse, tags=["LLM + RAG"])
def rag_similar(req: SimilarRequest):
    with _timed_span(req.trace_id, "recommender.rag_similar", dataset=req.dataset, top_k=req.top_k):
        if req.dataset not in SUPPORTED_DATASETS:
            raise HTTPException(400, f"Unsupported dataset: {req.dataset}")

        items_df = _load_items(req.dataset)
        item_id_col = _get_item_id_col(items_df)
        title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
        mask = items_df[title_col].astype(str).str.contains(req.item_title, case=False, na=False, regex=False)
        matches = items_df[mask]
        if matches.empty:
            raise HTTPException(404, f"Item '{req.item_title}' not found in dataset '{req.dataset}'")

        seed_row = matches.iloc[0]
        seed_title = str(seed_row[title_col])
        seed_item_id = str(seed_row[item_id_col])

        try:
            source = _rag_seed_item_source(req.dataset, seed_item_id)
        except Exception as exc:
            raise HTTPException(
                404,
                f"Run not trained: LLM + RAG semantic index is not available for dataset '{req.dataset}'.",
            ) from exc

        embedding_value = source.get("embedding")
        if isinstance(embedding_value, list) and embedding_value:
            query_vector = np.asarray(embedding_value, dtype=np.float32)
        else:
            text_repr = str(source.get("text_repr", "")).strip() or seed_title
            embedding_model = _get_sentence_transformer(
                os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            )
            query_vector = np.asarray(
                embedding_model.encode(text_repr, show_progress_bar=False),
                dtype=np.float32,
            )

        candidates = _rag_knn_candidates_from_vector(
            req.dataset,
            query_vector,
            req.top_k + 1,
            excluded_item_ids={seed_item_id},
        )
        item_ids = [item_id for item_id, _ in candidates[: req.top_k]]
        similar_items = _similar_items_from_item_ids(req.dataset, item_ids, req.top_k)
        return SimilarResponse(
            seed_title=seed_title,
            dataset=req.dataset,
            model="llm_rag",
            items=similar_items,
            trace_id=req.trace_id,
        )

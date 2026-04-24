"""
Recommender Microservice
========================
FastAPI service with model-specific endpoints.

Endpoints:
  GET  /health
  POST /search          →  Fuzzy text search in items.parquet (no model needed)
  POST /mf              →  Matrix Factorization user recommendations
  POST /mf/similar      →  MF item-to-item similarity
  POST /two_tower       →  Two-Tower recommendations          (future)
  POST /two_tower/similar  →  Two-Tower item similarity       (future)
  POST /sasrec          →  SASRec sequential recommendations  (future)

Usage (run from pfg-app/recommender/):
  uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import logging
import re
import sys
import time
import unicodedata
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

# ── Paths ─────────────────────────────────────────────────────────────────────
# pfg-app/recommender/main.py  →  parents[2] = pfg/
REPO_ROOT  = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from shared.contracts import (  # noqa: E402
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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[1]


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


def _prompt_tokens(prompt: str) -> list[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "into", "about", "this", "that",
        "want", "would", "like", "please", "recommend", "recommendation",
        "movie", "movies", "film", "films", "show", "shows", "something",
    }
    tokens = _tokenize_text(prompt)
    return [t for t in tokens if len(t) >= 4 and t not in stopwords]


def _genre_targets_from_prompt(prompt: str) -> set[str]:
    text = _normalize_text(prompt)
    genre_aliases = {
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
    targets: set[str] = set()
    for canonical, aliases in genre_aliases.items():
        if any(alias in text for alias in aliases):
            targets.add(canonical)
    return targets


def _filter_candidates_by_genres(dataset: str, candidate_indices: list[int], genres: set[str]) -> list[int]:
    if not genres:
        return candidate_indices
    items_df = _load_items(dataset)
    genre_col = _get_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])
    filtered: list[int] = []
    for idx in candidate_indices:
        if idx >= len(items_df):
            continue
        row_genres = _normalize_text(str(items_df.iloc[idx].get(genre_col, "")))
        if any(g in row_genres for g in genres):
            filtered.append(idx)
    return filtered


def _rerank_by_prompt(dataset: str, candidate_indices: list[int], prompt: str, top_k: int) -> list[int]:
    tokens = _prompt_tokens(prompt)
    if not tokens:
        return candidate_indices[:top_k]

    items_df = _load_items(dataset)
    title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
    genre_col = _get_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])

    scored: list[tuple[float, int]] = []
    for rank, idx in enumerate(candidate_indices):
        if idx >= len(items_df):
            continue
        row = items_df.iloc[idx]
        title = _normalize_text(str(row.get(title_col, "")))
        genres = _normalize_text(str(row.get(genre_col, "")))

        title_hits = sum(1 for t in tokens if t in title)
        genre_hits = sum(1 for t in tokens if t in genres)
        base = max(0.0, 1.0 - rank / max(len(candidate_indices), 1))
        score = base + 0.75 * title_hits + 1.5 * genre_hits
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
    genre_col = _get_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])

    recs = []
    for rank, idx in enumerate(item_indices):
        if idx >= len(items_df):
            continue
        row = items_df.iloc[idx]
        recs.append(RecommendedItem(
            title=str(row.get(title_col, f"Item {idx}")),
            score=round(1.0 - rank / len(item_indices), 3),
            genres=str(row.get(genre_col, "")) if genre_col in items_df.columns else "",
        ))

    return RecommendResponse(
        user_id=user_id,
        dataset=dataset,
        model=model,
        cold_start=cold_start,
        items=recs,
        trace_id=trace_id,
    )


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


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "endpoints": ["/search", "/mf", "/mf/similar", "/two_tower", "/sasrec"],
    }


# ── /search — Fuzzy text search in items.parquet ───────────────────────────────
@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search_items(req: SearchRequest):
    """Full-text (case-insensitive substring) search in the dataset's items metadata."""
    with _timed_span(req.trace_id, "recommender.search", dataset=req.dataset, top_k=req.top_k):
        if req.dataset not in SUPPORTED_DATASETS:
            raise HTTPException(400, f"Unsupported dataset: {req.dataset}")

        items_df = _load_items(req.dataset)
        title_col = _get_col(items_df, ["title", "name", "item_name", "business_name", "track_name"])
        genre_col = _get_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])

        mask = items_df[title_col].astype(str).str.contains(req.query, case=False, na=False)
        matches = items_df[mask].head(req.top_k)

        # Fallback semantic-ish token search for reordered titles (e.g., "The Matrix" vs "Matrix, The (1999)")
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

        results = [
            RecommendedItem(
                title=str(row[title_col]),
                score=1.0,
                genres=str(row.get(genre_col, "")) if genre_col in items_df.columns else "",
            )
            for _, row in matches.iterrows()
        ]
        log.info(
            "[SEARCH] trace_id=%r query=%r dataset=%r n_results=%s preview_titles=%s",
            req.trace_id,
            req.query,
            req.dataset,
            len(results),
            [r.title for r in results[:5]],
        )
        return SearchResponse(query=req.query, dataset=req.dataset, results=results, trace_id=req.trace_id)


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

        candidate_n = max(req.top_k * 20, 200)
        prompt_genres = _genre_targets_from_prompt(req.prompt)

        if user_factors_path.exists():
            log.info(
                "[MF] trace_id=%r user_id=%r dataset=%r external user, using popularity fallback",
                req.trace_id,
                req.user_id,
                req.dataset,
            )
            indices = _popularity_fallback(req.dataset, candidate_n)
            genre_filtered = _filter_candidates_by_genres(req.dataset, indices, prompt_genres)
            if len(genre_filtered) >= req.top_k:
                indices = genre_filtered
            indices = _rerank_by_prompt(req.dataset, indices, req.prompt, req.top_k)
            cold_start = True
        else:
            log.warning(
                "[MF] trace_id=%r no user_factors found for dataset=%r, using popularity fallback",
                req.trace_id,
                req.dataset,
            )
            indices = _popularity_fallback(req.dataset, candidate_n)
            genre_filtered = _filter_candidates_by_genres(req.dataset, indices, prompt_genres)
            if len(genre_filtered) >= req.top_k:
                indices = genre_filtered
            indices = _rerank_by_prompt(req.dataset, indices, req.prompt, req.top_k)

        response = _build_response(
            req.user_id,
            req.dataset,
            "matrix_factorization",
            indices,
            cold_start,
            trace_id=req.trace_id,
        )
        log.info(
            "[MF] trace_id=%r prompt=%r tokens=%s genre_targets=%s",
            req.trace_id,
            req.prompt,
            _prompt_tokens(req.prompt),
            sorted(prompt_genres),
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
        genre_col = _get_col(items_df, ["genres_str", "category", "genres", "tags", "categories"])

        mask = items_df[title_col].astype(str).str.contains(req.item_title, case=False, na=False)
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
                genres=str(row.get(genre_col, "")) if genre_col in items_df.columns else "",
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
@app.post("/two_tower", tags=["Two-Tower"])
def two_tower_recommend(req: RecommendRequest):
    raise HTTPException(501, "Two-Tower endpoint not yet implemented. Use /mf for now.")


# ── /sasrec — placeholder ──────────────────────────────────────────────────────
@app.post("/sasrec", tags=["SASRec"])
def sasrec_recommend(req: RecommendRequest):
    raise HTTPException(501, "SASRec endpoint not yet implemented. Use /mf for now.")

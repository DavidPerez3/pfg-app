from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


IntentName = Literal["user_recommendation", "entity_lookup", "item_similarity", "general_qa"]


class IntentAttributes(BaseModel):
    item_query: str = ""
    needs_weather_tool: bool = False
    reason: str = "ok"


class IntentClassification(BaseModel):
    intent: IntentName = "general_qa"
    attributes: IntentAttributes = Field(default_factory=IntentAttributes)


class RecommendedItem(BaseModel):
    title: str
    score: float
    genres: str = ""


class RecommendRequest(BaseModel):
    user_id: str
    dataset: str = "movielens"
    prompt: str = ""
    top_k: int = 10
    trace_id: str | None = None
    origin_intent: IntentName | None = None


class RecommendResponse(BaseModel):
    user_id: str
    dataset: str
    model: str
    cold_start: bool
    items: list[RecommendedItem]
    trace_id: str | None = None


class SearchRequest(BaseModel):
    query: str
    dataset: str = "movielens"
    top_k: int = 10
    trace_id: str | None = None


class SearchResponse(BaseModel):
    query: str
    dataset: str
    results: list[RecommendedItem]
    trace_id: str | None = None


class SimilarRequest(BaseModel):
    item_title: str
    dataset: str = "movielens"
    top_k: int = 10
    trace_id: str | None = None


class SimilarResponse(BaseModel):
    seed_title: str
    dataset: str
    model: str
    items: list[RecommendedItem]
    trace_id: str | None = None


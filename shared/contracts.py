from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


IntentName = Literal["user_recommendation", "entity_lookup", "item_similarity", "general_qa"]
ResultKind = Literal["recommendations", "search_results", "similar_items"]


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


class LatencyTelemetry(BaseModel):
    client_to_backend_ms: float | None = None
    backend_total_ms: float | None = None
    backend_to_recommender_http_ms: float | None = None
    recommender_total_ms: float | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    thread_id: str | None = None
    user_id: str | None = None
    trace_id: str | None = None
    client_sent_at_ms: float | None = None
    dataset: str | None = None
    rec_model: str | None = None
    dataset_user_id: str | None = None


class ChatResult(BaseModel):
    kind: ResultKind
    title: str
    subtitle: str = ""
    items: list[RecommendedItem] = Field(default_factory=list)
    dataset: str | None = None
    rec_model: str | None = None
    dataset_user_id: str | None = None
    cold_start: bool | None = None
    query: str | None = None
    seed_title: str | None = None
    trace_id: str | None = None
    explanation: str | None = None
    follow_up_prompts: list[str] = Field(default_factory=list)
    latency: LatencyTelemetry | None = None


class ChatResponse(BaseModel):
    messages: list[ChatMessage]
    trace_id: str | None = None
    intent: IntentName | None = None
    dataset: str | None = None
    rec_model: str | None = None
    result: ChatResult | None = None
    latency: LatencyTelemetry | None = None


class RecommendRequest(BaseModel):
    user_id: str
    dataset: str = "movielens"
    prompt: str = ""
    top_k: int = 10
    trace_id: str | None = None
    origin_intent: IntentName | None = None
    dataset_user_id: str | None = None


class RecommendResponse(BaseModel):
    user_id: str
    dataset: str
    model: str
    cold_start: bool
    items: list[RecommendedItem]
    trace_id: str | None = None
    explanation: str | None = None
    latency: LatencyTelemetry | None = None


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
    latency: LatencyTelemetry | None = None


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
    latency: LatencyTelemetry | None = None


class FeedbackRequest(BaseModel):
    user_id: str
    thread_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""
    message_index: int | None = None
    trace_id: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str


class FeedbackSummary(BaseModel):
    count: int
    average_rating: float | None = None
    distribution: dict[str, int] = Field(default_factory=dict)


class MemoryFact(BaseModel):
    fact: str
    source: str = "app"
    created_at: str


class MemoryResponse(BaseModel):
    user_id: str
    facts: list[MemoryFact] = Field(default_factory=list)
    count: int = 0


class SessionMemoryMessage(BaseModel):
    role: str
    content: str
    created_at: str


class ShortTermMemoryResponse(BaseModel):
    thread_id: str
    messages: list[SessionMemoryMessage] = Field(default_factory=list)
    count: int = 0
    window_hours: int = 24


class DatasetUserOption(BaseModel):
    user_id: str
    interaction_count: int


class DatasetUsersResponse(BaseModel):
    dataset: str
    users: list[DatasetUserOption] = Field(default_factory=list)
    total_available: int = 0
    latency: LatencyTelemetry | None = None

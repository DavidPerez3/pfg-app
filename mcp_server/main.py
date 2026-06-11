from __future__ import annotations

import json
import re
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse


DATASETS = [
    {
        "id": "movielens",
        "label": "MovieLens",
        "domain": "movies",
        "feedback": "explicit",
        "note": "Best interactive lookup/support dataset together with Yelp.",
    },
    {
        "id": "amazon_electronics",
        "label": "Amazon Electronics",
        "domain": "e-commerce",
        "feedback": "explicit",
        "note": "Usable, but the source catalog remains title-poor in several cases.",
    },
    {
        "id": "yelp",
        "label": "Yelp",
        "domain": "local business",
        "feedback": "explicit",
        "note": "Metadata-rich and one of the strongest user-facing datasets.",
    },
    {
        "id": "lastfm",
        "label": "LastFM",
        "domain": "music",
        "feedback": "implicit",
        "note": "Final benchmark interpretation is artist-level, not track-level.",
    },
    {
        "id": "foursquare",
        "label": "Foursquare",
        "domain": "mobility / check-ins",
        "feedback": "implicit",
        "note": "Recommendation serving works, but lookup metadata are structurally weak.",
    },
]

MODELS = [
    {
        "id": "matrix-factorization",
        "label": "Matrix Factorization",
        "family": "collaborative filtering",
        "note": "Fastest and strongest all-round benchmark baseline.",
    },
    {
        "id": "two-tower",
        "label": "Two-Tower",
        "family": "neural retrieval",
        "note": "Best primary ranking family in MovieLens.",
    },
    {
        "id": "two-tower-wide-deep",
        "label": "Two-Tower + Wide&Deep",
        "family": "two-stage retrieval + reranking",
        "note": "Most useful as a costlier trade-off architecture, not as the default winner.",
    },
    {
        "id": "sasrec",
        "label": "SASRec",
        "family": "sequence-aware recommendation",
        "note": "Usually strongest in diversity/coverage rather than top-k ranking.",
    },
    {
        "id": "llm-rag",
        "label": "LLM + RAG",
        "family": "semantic retrieval + optional reranking",
        "note": "Architecturally valuable and provider-aware, but not the strongest offline ranker.",
    },
]

ARCHITECTURE_FACTS = {
    "gateway": "The runtime is split into frontend, backend, recommender, PostgreSQL, Elasticsearch, and optional Kibana.",
    "backend": "The FastAPI backend acts as the application gateway and owns thread recovery, routing, memory access, feedback capture, and response normalization.",
    "recommender": "The recommender is a separate model-serving microservice exposing versioned recommendation, similarity, search, dataset-user, and health endpoints.",
    "memory": "Short-term memory is thread/session history. Long-term memory stores extracted user facts in SQL and retrieves them semantically through Elasticsearch.",
    "deployment": "The deployment-ready stack is rooted at pfg-stack, with nginx as the reverse-proxy entrypoint and pfg-models kept outside the long-lived runtime.",
    "providers": "The final runtime is Gemini-first in both the backend conversational path and the recommender LLM+RAG path, while Ollama remains an optional fallback/development runtime.",
    "mcp": "The current MCP implementation is intentionally minimal: one MCP server exposes structured project/runtime knowledge through standard MCP resources and tools, and the backend consumes it as an MCP client for project-capability questions.",
}

RESOURCE_PAYLOADS = {
    "pfg://datasets": {"datasets": DATASETS},
    "pfg://models": {"models": MODELS},
    "pfg://architecture": {"facts": ARCHITECTURE_FACTS},
}

TOPIC_KEYWORDS = {
    "datasets": ["dataset", "datasets", "movielens", "yelp", "lastfm", "foursquare", "amazon"],
    "models": ["model", "models", "matrix factorization", "two-tower", "wide", "deep", "sasrec", "rag"],
    "deployment": ["deploy", "deployment", "docker", "nginx", "no-ip", "server", "stack", "postgres", "kibana"],
    "memory": ["memory", "feedback", "thread", "session", "long-term", "short-term"],
    "architecture": ["architecture", "backend", "frontend", "recommender", "microservice", "api", "gateway"],
    "providers": ["gemini", "ollama", "provider", "llm runtime"],
    "mcp": ["mcp", "model context protocol", "tool", "resource", "prompt"],
}


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _detect_topic(question: str) -> str:
    normalized = _normalize(question)
    if not normalized:
        return "architecture"

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", normalized):
                return topic
    return "architecture"


def _render_dataset_answer() -> str:
    lines = ["Supported benchmark datasets:"]
    for dataset in DATASETS:
        lines.append(
            f"- {dataset['label']} ({dataset['id']}): {dataset['domain']}, {dataset['feedback']} feedback. {dataset['note']}"
        )
    return "\n".join(lines)


def _render_model_answer() -> str:
    lines = ["Supported recommendation paradigms:"]
    for model in MODELS:
        lines.append(f"- {model['label']} ({model['id']}): {model['family']}. {model['note']}")
    return "\n".join(lines)


def _render_architecture_answer() -> str:
    return "\n".join(
        [
            "Current runtime architecture:",
            f"- {ARCHITECTURE_FACTS['gateway']}",
            f"- {ARCHITECTURE_FACTS['backend']}",
            f"- {ARCHITECTURE_FACTS['recommender']}",
        ]
    )


def _render_deployment_answer() -> str:
    return "\n".join(
        [
            "Deployment shape:",
            f"- {ARCHITECTURE_FACTS['deployment']}",
            "- Elasticsearch is mandatory in runtime for entity lookup, long-term memory retrieval, and LLM+RAG.",
            "- PostgreSQL is the intended application-state store.",
            "- Kibana is optional and admin-only.",
        ]
    )


def _render_memory_answer() -> str:
    return "\n".join(
        [
            "Memory and refinement design:",
            f"- {ARCHITECTURE_FACTS['memory']}",
            "- Feedback is stored explicitly and can steer later recommendation refinements.",
        ]
    )


def _render_provider_answer() -> str:
    return "\n".join(
        [
            "LLM provider strategy:",
            f"- {ARCHITECTURE_FACTS['providers']}",
        ]
    )


def _render_mcp_answer() -> str:
    return "\n".join(
        [
            "Minimal MCP implementation:",
            f"- {ARCHITECTURE_FACTS['mcp']}",
            "- This server exposes MCP resources for datasets, models, and architecture facts, plus one MCP tool for question answering over that structured project knowledge.",
        ]
    )


def _answer_for_topic(topic: str) -> str:
    if topic == "datasets":
        return _render_dataset_answer()
    if topic == "models":
        return _render_model_answer()
    if topic == "deployment":
        return _render_deployment_answer()
    if topic == "memory":
        return _render_memory_answer()
    if topic == "providers":
        return _render_provider_answer()
    if topic == "mcp":
        return _render_mcp_answer()
    return _render_architecture_answer()


def _capabilities_payload() -> dict[str, Any]:
    return {
        "service": "pfg-benchmark-mcp",
        "resources": list(RESOURCE_PAYLOADS.keys()),
        "tools": ["answer_project_question"],
        "topics": list(TOPIC_KEYWORDS.keys()),
    }


mcp = FastMCP("PFG Benchmark MCP")


@mcp.resource("pfg://datasets")
def datasets_resource() -> str:
    return json.dumps(RESOURCE_PAYLOADS["pfg://datasets"], ensure_ascii=False, indent=2)


@mcp.resource("pfg://models")
def models_resource() -> str:
    return json.dumps(RESOURCE_PAYLOADS["pfg://models"], ensure_ascii=False, indent=2)


@mcp.resource("pfg://architecture")
def architecture_resource() -> str:
    return json.dumps(RESOURCE_PAYLOADS["pfg://architecture"], ensure_ascii=False, indent=2)


@mcp.tool()
def answer_project_question(question: str) -> dict[str, Any]:
    topic = _detect_topic(question)
    answer = _answer_for_topic(topic)
    return {
        "topic": topic,
        "question": question,
        "answer": answer,
    }


app = mcp.streamable_http_app()


async def health(_request: Request) -> JSONResponse:
    payload = _capabilities_payload()
    return JSONResponse(
        {
            "status": "ok",
            **payload,
        }
    )


async def capabilities(_request: Request) -> JSONResponse:
    return JSONResponse(_capabilities_payload())


app.add_route("/health", health, methods=["GET"])
app.add_route("/capabilities", capabilities, methods=["GET"])

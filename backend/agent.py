import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional dependency in some environments
    ChatGoogleGenerativeAI = None
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

try:
    import instructor
    from openai import OpenAI
except Exception:  # pragma: no cover - optional until dependency is installed
    instructor = None
    OpenAI = None


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config import settings  # noqa: E402
from hitl_utils import build_follow_up_prompts  # noqa: E402
from mcp_bridge import fetch_project_context_sync, looks_like_project_context_question  # noqa: E402
from shared.contracts import IntentClassification, RecommendRequest  # noqa: E402


class AgentState(TypedDict):
    """The LangGraph state for the conversational recommender."""

    messages: Annotated[list, add_messages]
    intent: str
    attributes: dict[str, Any]
    trace_id: str
    result: dict[str, Any]


RECOMMENDER_BASE_URL = settings.recommender_base_url
RECOMMENDER_API_V1_PREFIX = "/api/v1"

# Maps UI model selector values to canonical recommender model slugs.
MODEL_SLUG_MAP = {
    "mf": "matrix-factorization",
    "matrix_factorization": "matrix-factorization",
    "two_tower": "two-tower",
    "two_tower_wide_deep": "two-tower-wide-deep",
    "sasrec": "sasrec",
    "llm_rag": "llm-rag",
}

# Maps UI model selector values to recommender recommendation paths.
MODEL_ENDPOINT_MAP = {
    model_key: f"{RECOMMENDER_API_V1_PREFIX}/recommenders/{model_slug}/recommendations"
    for model_key, model_slug in MODEL_SLUG_MAP.items()
}

SUPPORTED_DATASETS = {"movielens", "amazon_electronics", "yelp", "lastfm", "foursquare"}
SUPPORTED_REC_MODELS = set(MODEL_ENDPOINT_MAP.keys())
VALID_INTENTS = {"user_recommendation", "entity_lookup", "item_similarity", "general_qa"}
INTENT_ALIASES = {
    "recommendation": "user_recommendation",
    "recommend": "user_recommendation",
    "recommendations": "user_recommendation",
    "userrecommendation": "user_recommendation",
    "lookup": "entity_lookup",
    "entitylookup": "entity_lookup",
    "search": "entity_lookup",
    "find": "entity_lookup",
    "similar": "item_similarity",
    "similarity": "item_similarity",
    "itemsimilarity": "item_similarity",
    "qa": "general_qa",
    "general": "general_qa",
    "question_answering": "general_qa",
}
GENRE_THEME_TERMS = {
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "horror",
    "musical",
    "mystery",
    "noir",
    "romance",
    "romantic",
    "sci fi",
    "sci-fi",
    "scifi",
    "science fiction",
    "thriller",
    "thrillers",
    "war",
    "western",
}
PREFERENCE_STATEMENT_PREFIXES = (
    "i like ",
    "i love ",
    "i prefer ",
    "i enjoy ",
    "my favorite",
    "my favourite",
    "i don't like ",
    "i do not like ",
    "i hate ",
)


def _backend_llm_provider() -> str:
    provider = (settings.backend_llm_provider or "gemini").strip().lower()
    return provider if provider in {"gemini", "ollama"} else "gemini"


def _backend_llm_model_name(provider: str | None = None) -> str:
    chosen = provider or _backend_llm_provider()
    return settings.backend_gemini_model if chosen == "gemini" else settings.backend_ollama_model


def _gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini in the backend chatbot.")
    return api_key


def _build_backend_chat_llm(temperature: float = 0, tools: list | None = None):
    provider = _backend_llm_provider()
    if provider == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain-google-genai is not installed for Gemini backend execution.")
        llm = ChatGoogleGenerativeAI(
            model=_backend_llm_model_name(provider),
            google_api_key=_gemini_api_key(),
            temperature=temperature,
        )
        return llm.bind_tools(tools) if tools else llm

    llm = ChatOllama(model=_backend_llm_model_name(provider), temperature=temperature)
    return llm.bind_tools(tools) if tools else llm


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    cfg = (config or {}).get("configurable", {})
    return cfg if isinstance(cfg, dict) else {}


def _resolve_trace_id(state: AgentState | dict[str, Any], config: RunnableConfig | None) -> str:
    from_state = state.get("trace_id") if isinstance(state, dict) else None
    if isinstance(from_state, str) and from_state.strip():
        return from_state

    cfg = _configurable(config)
    from_cfg = cfg.get("trace_id")
    if isinstance(from_cfg, str) and from_cfg.strip():
        return from_cfg.strip()

    thread_id = cfg.get("thread_id")
    if isinstance(thread_id, str) and thread_id.strip():
        return f"thread-{thread_id}"
    return f"trace-{uuid.uuid4().hex[:12]}"


def _resolve_dataset(config: RunnableConfig | None) -> str:
    cfg = _configurable(config)
    dataset = cfg.get("dataset")
    if isinstance(dataset, str) and dataset in SUPPORTED_DATASETS:
        return dataset
    return settings.default_dataset


def _resolve_rec_model(config: RunnableConfig | None) -> str:
    cfg = _configurable(config)
    rec_model = cfg.get("rec_model")
    if isinstance(rec_model, str) and rec_model in SUPPORTED_REC_MODELS:
        return rec_model
    return settings.default_rec_model


def _resolve_user_id(config: RunnableConfig | None) -> str:
    cfg = _configurable(config)
    user_id = cfg.get("user_id")
    if isinstance(user_id, str) and user_id.strip():
        return user_id.strip()
    return "anonymous"


def _resolve_dataset_user_id(config: RunnableConfig | None) -> str | None:
    cfg = _configurable(config)
    dataset_user_id = cfg.get("dataset_user_id")
    if isinstance(dataset_user_id, str) and dataset_user_id.strip():
        return dataset_user_id.strip()
    return None


def _resolve_hitl_refinement_active(config: RunnableConfig | None) -> bool:
    cfg = _configurable(config)
    return bool(cfg.get("hitl_refinement_active"))


def _resolve_hitl_refinement_context(config: RunnableConfig | None) -> str | None:
    cfg = _configurable(config)
    value = cfg.get("hitl_refinement_context")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _contains_genre_or_theme_term(normalized: str) -> bool:
    if not normalized:
        return False
    for term in GENRE_THEME_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            return True
    return False


def _looks_like_theme_recommendation_prompt(normalized: str) -> bool:
    if not normalized:
        return False

    if not _contains_genre_or_theme_term(normalized):
        return False

    explicit_preference_markers = (
        "i like ",
        "i love ",
        "i prefer ",
        "i enjoy ",
        "i'm into ",
        "im into ",
        "looking for ",
        "in the mood for ",
        "give me ",
        "show me some ",
        "want some ",
    )
    broad_theme_markers = (
        " and ",
        " or ",
        "more ",
        "less ",
        "without ",
        "with more ",
        "with less ",
    )

    return (
        normalized.startswith(explicit_preference_markers)
        or any(marker in normalized for marker in broad_theme_markers)
        or len(normalized.split()) <= 5
    )


def _classify_intent_fast_path(user_text: str) -> tuple[str, dict[str, Any]] | None:
    normalized = " ".join(user_text.lower().strip().split())
    if not normalized:
        return None

    recommendation_markers = (
        "recommend",
        "suggest",
        "what should i watch",
        "what should i listen",
        "what should i buy",
    )
    similarity_markers = ("similar to", "like ", "items like", "more like")
    lookup_markers = ("find ", "search ", "look up", "show me", "who is", "what is")
    if looks_like_project_context_question(user_text):
        return (
            "general_qa",
            {"item_query": "", "needs_weather_tool": False, "reason": "heuristic_project_context"},
        )

    if any(marker in normalized for marker in recommendation_markers):
        return (
            "user_recommendation",
            {"item_query": "", "needs_weather_tool": False, "reason": "heuristic_recommendation"},
        )

    if _looks_like_theme_recommendation_prompt(normalized):
        return (
            "user_recommendation",
            {"item_query": "", "needs_weather_tool": False, "reason": "heuristic_theme_recommendation"},
        )

    if normalized.startswith(PREFERENCE_STATEMENT_PREFIXES):
        return (
            "general_qa",
            {"item_query": "", "needs_weather_tool": False, "reason": "heuristic_preference_statement"},
        )

    if any(marker in normalized for marker in similarity_markers):
        item_query = user_text.strip()
        for marker in ("similar to", "like ", "items like", "more like"):
            if marker in normalized:
                start = normalized.find(marker) + len(marker)
                item_query = user_text[start:].strip(" .?!\"'")
                break
        return (
            "item_similarity",
            {"item_query": item_query, "needs_weather_tool": False, "reason": "heuristic_similarity"},
        )

    if any(marker in normalized for marker in lookup_markers):
        return (
            "entity_lookup",
            {"item_query": user_text.strip(), "needs_weather_tool": False, "reason": "heuristic_lookup"},
        )

    return None


def _kv(**kwargs: object) -> str:
    return " ".join(f"{k}={v!r}" for k, v in kwargs.items() if v is not None)


@contextmanager
def _timed_span(trace_id: str, stage: str, **kwargs: object):
    start = time.perf_counter()
    log.info("[TRACE][START] trace_id=%r stage=%r %s", trace_id, stage, _kv(**kwargs))
    try:
        yield
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        log.info("[TRACE][END] trace_id=%r stage=%r elapsed_ms=%s", trace_id, stage, elapsed_ms)


def _get_instructor_client():
    if instructor is None or OpenAI is None:
        raise RuntimeError("instructor/openai dependencies are not available")
    if _backend_llm_provider() != "ollama":
        raise RuntimeError("instructor fast-path is only configured for the Ollama backend")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    openai_client = OpenAI(base_url=base_url, api_key=api_key)
    return instructor.from_openai(openai_client, mode=instructor.Mode.JSON)


def _classify_intent_with_instructor(user_text: str) -> IntentClassification:
    model_name = settings.backend_ollama_model
    client = _get_instructor_client()
    return client.chat.completions.create(
        model=model_name,
        temperature=0,
        response_model=IntentClassification,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a workflow router for a recommender chatbot. "
                    "Return structured output only."
                ),
            },
            {
                "role": "user",
                "content": (
                "Classify the latest user message.\n"
                "Valid intents: user_recommendation, entity_lookup, item_similarity, general_qa.\n"
                "If the user expresses genres, themes, moods, or broad preferences without a concrete title/entity, classify as user_recommendation.\n"
                "Use entity_lookup or item_similarity only when the message refers to a specific title, item, artist, product, or place.\n"
                "For attributes.item_query extract only the core entity/title when relevant.\n"
                "Set attributes.needs_weather_tool=true only if weather is explicitly requested.\n"
                f"Message: {user_text}"
            ),
        },
        ],
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts).strip()
    return str(content)


def _parse_json_from_model_output(raw: Any) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        # Try to recover when the model wraps JSON in extra prose.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _validate_classifier_payload(payload: Any) -> tuple[str, dict[str, Any]]:
    default = IntentClassification()
    if not isinstance(payload, dict):
        return default.intent, default.attributes.model_dump()

    def _coerce_classifier_intent(value: Any) -> str:
        if not isinstance(value, str):
            return default.intent
        normalized = re.sub(r"[\s\-]+", "_", value.strip().lower())
        normalized = re.sub(r"[^a-z_]", "", normalized)
        if normalized in VALID_INTENTS:
            return normalized
        return INTENT_ALIASES.get(normalized, default.intent)

    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "weather", "required"}:
                return True
            if normalized in {"false", "0", "no", "n", "", "none", "null"}:
                return False
        return default.attributes.needs_weather_tool

    def _clean_model_string(value: Any, *, fallback: str = "", max_len: int = 500) -> str:
        if value is None:
            return fallback
        if isinstance(value, (dict, list, tuple, set)):
            return fallback
        cleaned = " ".join(str(value).strip().split())
        if cleaned.lower() in {"none", "null", "n/a", "na", "unknown"}:
            return fallback
        return cleaned[:max_len]

    raw_attributes = payload.get("attributes")
    attrs = raw_attributes if isinstance(raw_attributes, dict) else {}
    item_query = _clean_model_string(
        attrs.get("item_query", payload.get("item_query")),
        fallback="",
        max_len=300,
    )
    reason = _clean_model_string(
        attrs.get("reason", payload.get("reason")),
        fallback=default.attributes.reason,
        max_len=120,
    ) or default.attributes.reason
    needs_weather_tool = _coerce_bool(
        attrs.get("needs_weather_tool", payload.get("needs_weather_tool", default.attributes.needs_weather_tool))
    )

    normalized_payload = {
        "intent": _coerce_classifier_intent(payload.get("intent")),
        "attributes": {
            "item_query": item_query,
            "needs_weather_tool": needs_weather_tool,
            "reason": reason,
        },
    }

    try:
        parsed = IntentClassification.model_validate(normalized_payload)
    except Exception:
        return default.intent, default.attributes.model_dump()
    return parsed.intent, parsed.attributes.model_dump()


def _safe_item_query_from_attributes(attrs: dict[str, Any], fallback_prompt: str) -> str:
    candidate = attrs.get("item_query")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return fallback_prompt.strip()


def _clean_entity_lookup_query(query: str) -> str:
    cleaned = " ".join(str(query or "").strip().split())
    if not cleaned:
        return ""

    patterns = (
        r"^(find|search)\s+",
        r"^(look up)\s+",
        r"^(show me)\s+",
        r"^(who is)\s+",
        r"^(what is)\s+",
    )
    for pattern in patterns:
        cleaned = __import__("re").sub(pattern, "", cleaned, flags=__import__("re").IGNORECASE)
        cleaned = cleaned.strip(" .?!\"'")
    return cleaned.strip()


def _sanitize_intent_classification(
    user_text: str,
    intent: str,
    attributes: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    normalized = " ".join((user_text or "").lower().strip().split())
    attrs = dict(attributes or {})
    item_query = str(attrs.get("item_query") or "").strip()

    if looks_like_project_context_question(user_text):
        return (
            "general_qa",
            {
                "item_query": "",
                "needs_weather_tool": False,
                "reason": "sanitized_project_context_question",
            },
        )

    if intent in {"entity_lookup", "item_similarity"}:
        cleaned_query = _clean_entity_lookup_query(item_query or user_text)
        cleaned_normalized = " ".join(cleaned_query.lower().split())

        if _looks_like_theme_recommendation_prompt(normalized) and _contains_genre_or_theme_term(
            cleaned_normalized
        ):
            return (
                "user_recommendation",
                {
                    "item_query": "",
                    "needs_weather_tool": bool(attrs.get("needs_weather_tool", False)),
                    "reason": "sanitized_theme_recommendation",
                },
            )

        if intent == "entity_lookup":
            attrs["item_query"] = cleaned_query
        elif not item_query and cleaned_query:
            attrs["item_query"] = cleaned_query

    return intent, attrs


def _message_type(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").strip().lower()
    return str(getattr(message, "type", "") or "").strip().lower()


def _collect_recommendation_supporting_context(
    messages: list[Any],
    *,
    include_refinement_context: bool,
) -> list[str]:
    contexts: list[str] = []
    preference_hints: list[str] = []

    for message in messages[:-1]:
        msg_type = _message_type(message)
        text = _content_to_text(message.get("content", "")) if isinstance(message, dict) else _content_to_text(getattr(message, "content", ""))
        if not text:
            continue

        normalized = " ".join(text.lower().strip().split())
        if msg_type in {"system"}:
            if text.startswith("Persistent user memory facts."):
                contexts.append(text.strip())
                continue
            if text.startswith("Recommendation follow-up context for this thread:"):
                contexts.append(text.strip())
                continue
            if include_refinement_context and text.startswith("Human-in-the-loop refinement context for this thread:"):
                contexts.append(text.strip())
                continue

        if msg_type in {"human", "user"} and (
            normalized.startswith(PREFERENCE_STATEMENT_PREFIXES)
            or _looks_like_theme_recommendation_prompt(normalized)
        ):
            preference_hints.append(text.strip())

    if preference_hints:
        contexts.insert(
            0,
            "Previous user preference hints:\n" + "\n".join(f"- {hint}" for hint in preference_hints[-3:]),
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for context in contexts:
        key = context.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _extract_previous_recommendation_titles(messages: list[Any]) -> list[str]:
    for message in messages:
        if _message_type(message) != "system":
            continue
        text = _content_to_text(message.get("content", "")) if isinstance(message, dict) else _content_to_text(getattr(message, "content", ""))
        match = re.search(r"Top recommended items:\s*(.+?)(?:\.|$)", text)
        if not match:
            continue
        titles = [part.strip() for part in match.group(1).split(";") if part.strip()]
        if titles:
            return titles
    return []


def _extract_context_line(messages: list[Any], label: str) -> str:
    pattern = re.compile(rf"{re.escape(label)}\s*(.+)")
    for message in messages:
        if _message_type(message) != "system":
            continue
        text = _content_to_text(message.get("content", "")) if isinstance(message, dict) else _content_to_text(getattr(message, "content", ""))
        for line in text.splitlines():
            match = pattern.search(line.strip())
            if match:
                return match.group(1).strip()
    return ""


def _build_deterministic_recommendation_follow_up_answer(messages: list[Any], last_content: str) -> str | None:
    normalized = " ".join((last_content or "").lower().strip().split())
    if not normalized:
        return None

    titles = _extract_previous_recommendation_titles(messages)
    previous_request = _extract_context_line(messages, "Previous recommendation request:")
    feedback_note = _extract_context_line(messages, "Latest user feedback note:")
    if not feedback_note:
        feedback_note = _extract_context_line(messages, "Stored feedback note:")

    if not any([titles, previous_request, feedback_note]):
        return None

    starter_markers = (
        "which one should i start with",
        "which one should i start",
        "start with",
        "which one",
        "best one",
        "first one",
        "pick one",
    )
    why_markers = (
        "why these",
        "why this list",
        "explain these",
        "explain this list",
        "why did you recommend",
    )

    if any(marker in normalized for marker in starter_markers) and titles:
        top_title = titles[0]
        return (
            f"Start with {top_title} first. "
            "It is the safest entry point from the previous recommendation list because it was surfaced at the top of that set."
        )

    if any(marker in normalized for marker in why_markers):
        parts: list[str] = []
        if previous_request:
            parts.append(f"These recommendations are anchored to your earlier request: {previous_request}.")
        else:
            parts.append("These recommendations are anchored to your previous recommendation context.")
        if feedback_note:
            parts.append(f"I also kept your latest feedback note in mind: {feedback_note}.")
        if titles:
            parts.append(f"The previous list was headed by {titles[0]}.")
        parts.append("If you want, I can now compare the top items one by one.")
        return " ".join(parts)

    return None


def _is_recommendation_follow_up_question(messages: list[Any], last_content: str) -> bool:
    return _build_deterministic_recommendation_follow_up_answer(messages, last_content) is not None


def _extract_feedback_note(refinement_context: str | None) -> str:
    if not refinement_context:
        return ""
    match = re.search(r"Stored feedback note:\s*(.+)", refinement_context)
    if not match:
        return ""
    return match.group(1).strip()


def _is_generic_retry_prompt(prompt: str) -> bool:
    normalized = " ".join((prompt or "").lower().strip().split())
    if not normalized:
        return False
    generic_markers = (
        "another try",
        "try again",
        "retry",
        "another one",
        "something else",
        "different options",
        "different recommendations",
        "not these",
        "give me another",
        "start over",
    )
    return any(marker in normalized for marker in generic_markers)


def _build_effective_recommendation_prompt(last_prompt: str, refinement_context: str | None) -> str:
    if not refinement_context:
        return last_prompt

    current_request = last_prompt.strip()
    feedback_note = _extract_feedback_note(refinement_context)
    if feedback_note and _is_generic_retry_prompt(current_request):
        current_request = (
            f"{current_request}\n"
            f"Use this stored feedback to steer the reranking: {feedback_note}"
        )

    return f"{refinement_context}\n\nCurrent refinement request:\n{current_request}"


def _response_latency_payload(remote_ms: float, response_json: dict[str, Any]) -> dict[str, float]:
    service_latency = response_json.get("latency") if isinstance(response_json, dict) else None
    recommender_total_ms = None
    if isinstance(service_latency, dict):
        maybe_total = service_latency.get("recommender_total_ms")
        if isinstance(maybe_total, (int, float)):
            recommender_total_ms = round(float(maybe_total), 2)

    payload: dict[str, float] = {
        "backend_to_recommender_http_ms": round(float(remote_ms), 2),
    }
    if recommender_total_ms is not None:
        payload["recommender_total_ms"] = recommender_total_ms
    return payload


def _extract_item_query_with_llm(user_prompt: str, instruction: str) -> str:
    llm = _build_backend_chat_llm(temperature=0)
    system_prompt = SystemMessage(content=instruction)
    human_prompt = HumanMessage(content=user_prompt)
    return str(llm.invoke([system_prompt, human_prompt]).content).strip()


@tool
def get_weather(location: str) -> str:
    """Fetch current weather for a location using Open-Meteo."""
    import requests
    import urllib3

    urllib3.disable_warnings()
    weather_trace_id = f"tool-{uuid.uuid4().hex[:8]}"
    with _timed_span(weather_trace_id, "tool.get_weather", location=location):
        try:
            geocode_url = (
                "https://geocoding-api.open-meteo.com/v1/search"
                f"?name={location}&count=1&language=en&format=json"
            )
            geo_res = requests.get(geocode_url, timeout=10, verify=False)
            geo_res.raise_for_status()
            geo_data = geo_res.json()

            if "results" not in geo_data:
                return f"Could not find coordinates for: {location}"

            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            name = geo_data["results"][0]["name"]

            weather_url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&current=temperature_2m,wind_speed_10m&timezone=auto"
            )
            weather_res = requests.get(weather_url, timeout=10, verify=False)
            weather_res.raise_for_status()
            w_data = weather_res.json()

            current = w_data["current"]
            temp = current["temperature_2m"]
            temp_units = w_data["current_units"]["temperature_2m"]
            return f"Current weather in {name}: {temp}{temp_units}."
        except Exception as exc:
            return f"Error fetching weather for {location}: {exc}"


def classifier_node(state: AgentState, config: RunnableConfig) -> dict:
    """LLM-driven workflow classifier and attribute extractor."""
    trace_id = _resolve_trace_id(state, config)
    messages = state.get("messages", [])
    if not messages:
        log.info("[CLASSIFIER] trace_id=%r no messages found, defaulting to general_qa", trace_id)
        default = IntentClassification()
        return {
            "trace_id": trace_id,
            "intent": default.intent,
            "attributes": default.attributes.model_dump(),
        }

    user_text = _content_to_text(messages[-1].content)

    with _timed_span(trace_id, "backend.classifier", preview=user_text[:80]):
        if _resolve_hitl_refinement_active(config):
            attributes = {
                "item_query": "",
                "needs_weather_tool": False,
                "reason": "thread_refinement",
            }
            log.info(
                "[CLASSIFIER] trace_id=%r via=hitl_refinement intent=%r attributes=%s",
                trace_id,
                "user_recommendation",
                attributes,
            )
            return {
                "trace_id": trace_id,
                "intent": "user_recommendation",
                "attributes": attributes,
            }

        if _is_recommendation_follow_up_question(messages, user_text):
            attributes = {
                "item_query": "",
                "needs_weather_tool": False,
                "reason": "recommendation_follow_up_question",
            }
            log.info(
                "[CLASSIFIER] trace_id=%r via=recommendation_follow_up intent=%r attributes=%s",
                trace_id,
                "general_qa",
                attributes,
            )
            return {
                "trace_id": trace_id,
                "intent": "general_qa",
                "attributes": attributes,
            }

        if looks_like_project_context_question(user_text):
            attributes = {
                "item_query": "",
                "needs_weather_tool": False,
                "reason": "project_context_question",
            }
            log.info(
                "[CLASSIFIER] trace_id=%r via=project_context intent=%r attributes=%s",
                trace_id,
                "general_qa",
                attributes,
            )
            return {
                "trace_id": trace_id,
                "intent": "general_qa",
                "attributes": attributes,
            }

        heuristic = _classify_intent_fast_path(user_text)
        if heuristic is not None:
            intent, attributes = heuristic
            intent, attributes = _sanitize_intent_classification(user_text, intent, attributes)
            log.info(
                "[CLASSIFIER] trace_id=%r via=heuristic intent=%r attributes=%s",
                trace_id,
                intent,
                attributes,
            )
            return {"trace_id": trace_id, "intent": intent, "attributes": attributes}

        try:
            parsed = _classify_intent_with_instructor(user_text)
            intent = parsed.intent
            attributes = parsed.attributes.model_dump()
            intent, attributes = _sanitize_intent_classification(user_text, intent, attributes)
            log.info(
                "[CLASSIFIER] trace_id=%r via=instructor intent=%r attributes=%s",
                trace_id,
                intent,
                attributes,
            )
            return {"trace_id": trace_id, "intent": intent, "attributes": attributes}
        except Exception as instructor_exc:
            log.warning(
                "[CLASSIFIER] trace_id=%r instructor classification failed, fallback to direct LLM parsing: %s",
                trace_id,
                instructor_exc,
            )

            try:
                llm = _build_backend_chat_llm(temperature=0)
                prompt = [
                    SystemMessage(
                        content=(
                            "You are a workflow router for a recommender chatbot. "
                            "Classify the latest user message and extract attributes. "
                            "If the user mentions only genres, themes, moods, or broad preferences without a concrete title/entity, use user_recommendation. "
                            "Use entity_lookup or item_similarity only for specific entities/titles. "
                            "Return ONLY valid JSON with this schema:\n"
                            "{\n"
                            '  "intent": "user_recommendation|entity_lookup|item_similarity|general_qa",\n'
                            '  "attributes": {\n'
                            '    "item_query": "string or empty",\n'
                            '    "needs_weather_tool": true|false,\n'
                            '    "reason": "short explanation"\n'
                            "  }\n"
                            "}"
                        )
                    ),
                    HumanMessage(content=user_text),
                ]
                raw = llm.invoke(prompt).content
                parsed_dict = _parse_json_from_model_output(raw)
                intent, attributes = _validate_classifier_payload(parsed_dict)
                intent, attributes = _sanitize_intent_classification(user_text, intent, attributes)
                log.info(
                    "[CLASSIFIER] trace_id=%r via=%s intent=%r attributes=%s",
                    trace_id,
                    _backend_llm_provider(),
                    intent,
                    attributes,
                )
                return {"trace_id": trace_id, "intent": intent, "attributes": attributes}
            except Exception as fallback_exc:
                log.warning(
                    "[CLASSIFIER] trace_id=%r fallback classification failed: %s",
                    trace_id,
                    fallback_exc,
                )
                default = IntentClassification()
                return {
                    "trace_id": trace_id,
                    "intent": default.intent,
                    "attributes": default.attributes.model_dump(),
                }

def user_recommendation_node(state: AgentState, config: RunnableConfig) -> dict:
    """Call recommender service and return deterministic recommendation text."""
    import requests as http_requests

    trace_id = _resolve_trace_id(state, config)
    dataset = _resolve_dataset(config)
    rec_model = _resolve_rec_model(config)
    user_id = _resolve_user_id(config)
    dataset_user_id = _resolve_dataset_user_id(config)
    refinement_context = _resolve_hitl_refinement_context(config)
    messages = state.get("messages", [])
    last_prompt = _content_to_text(messages[-1].content) if messages else ""
    effective_prompt = _build_effective_recommendation_prompt(last_prompt, refinement_context)
    supporting_contexts = _collect_recommendation_supporting_context(
        messages,
        include_refinement_context=not bool(refinement_context),
    )
    if supporting_contexts:
        effective_prompt = "\n\n".join(
            supporting_contexts + [f"Current recommendation request:\n{effective_prompt.strip()}"]
        )

    with _timed_span(
        trace_id,
        "backend.user_recommendation",
        dataset=dataset,
        model=rec_model,
        user_id=user_id,
        dataset_user_id=dataset_user_id,
    ):
        try:
            endpoint = MODEL_ENDPOINT_MAP.get(
                rec_model,
                f"{RECOMMENDER_API_V1_PREFIX}/recommenders/matrix-factorization/recommendations",
            )
            url = f"{RECOMMENDER_BASE_URL}{endpoint}"
            request_payload = RecommendRequest(
                user_id=str(user_id),
                dataset=str(dataset),
                prompt=effective_prompt,
                top_k=10,
                trace_id=trace_id,
                origin_intent="user_recommendation",
                dataset_user_id=dataset_user_id,
            )
            payload_dict = request_payload.model_dump()
            log.info("[USER_RECOMMENDATION] trace_id=%r POST %s payload=%s", trace_id, url, payload_dict)

            remote_start = time.perf_counter()
            recommendation_timeout = 90 if rec_model == "llm_rag" else 45
            resp = http_requests.post(
                url,
                json=payload_dict,
                timeout=recommendation_timeout,
                headers={"X-Trace-Id": trace_id},
            )
            remote_ms = round((time.perf_counter() - remote_start) * 1000, 2)
            resp.raise_for_status()

            data = resp.json()
            items = data.get("items", [])
            cold_start = data.get("cold_start", False)
            explanation = data.get("explanation")
            preview = [str(item.get("title", "")) for item in items[:5]]
            log.info(
                "[USER_RECOMMENDATION] trace_id=%r status=%s remote_ms=%s cold_start=%s n_items=%s preview_titles=%s",
                trace_id,
                resp.status_code,
                remote_ms,
                cold_start,
                len(items),
                preview,
            )

            if not items:
                return {"trace_id": trace_id, "messages": [AIMessage(content="No recommendations found for this profile yet.")]}

            cold_start_note = (
                " Cold-start mode."
                if cold_start
                else ""
            )
            content = (
                explanation
                or (
                    f"I found {len(items)} recommendations in `{dataset}` using `{data.get('model', rec_model)}`."
                    f"{cold_start_note} Review the ranked cards below."
                )
            )
            return {
                "trace_id": trace_id,
                "result": {
                    "kind": "recommendations",
                    "title": "Recommendations",
                    "subtitle": f"Dataset: {dataset} | Model: {data.get('model', rec_model)}",
                    "items": items,
                    "dataset": dataset,
                    "rec_model": data.get("model", rec_model),
                    "dataset_user_id": dataset_user_id,
                    "cold_start": cold_start,
                    "trace_id": trace_id,
                    "explanation": explanation,
                    "latency": _response_latency_payload(remote_ms, data),
                    "follow_up_prompts": build_follow_up_prompts(
                        items=items,
                        cold_start=bool(cold_start),
                    ),
                },
                "messages": [AIMessage(content=content)],
            }

        except http_requests.exceptions.ConnectionError:
            text = (
                "The recommender microservice is unavailable.\n"
                "Please make sure it is running at http://localhost:8001\n\n"
                f"-> Dataset: {dataset} | Model: {rec_model} | User: {user_id}"
            )
            return {"trace_id": trace_id, "messages": [AIMessage(content=text)]}
        except http_requests.exceptions.HTTPError as exc:
            detail = ""
            if exc.response is not None:
                try:
                    payload = exc.response.json()
                    detail = str(payload.get("detail", "")).strip()
                except Exception:
                    detail = exc.response.text.strip()

            if exc.response is not None and exc.response.status_code == 404 and detail:
                return {"trace_id": trace_id, "messages": [AIMessage(content=detail)]}

            message = detail or str(exc)
            return {"trace_id": trace_id, "messages": [AIMessage(content=f"Recommendation error: {message}")]}
        except Exception as exc:
            return {"trace_id": trace_id, "messages": [AIMessage(content=f"Recommendation error: {exc}")]}

def entity_lookup_node(state: AgentState, config: RunnableConfig) -> dict:
    """Search items/entities through the versioned dataset-item search endpoint."""
    import requests as http_requests

    trace_id = _resolve_trace_id(state, config)
    dataset = _resolve_dataset(config)
    last_prompt = _content_to_text(state["messages"][-1].content) if state.get("messages") else ""
    attrs = state.get("attributes", {})

    extracted_query = _safe_item_query_from_attributes(attrs, "")
    if not extracted_query:
        extracted_query = _extract_item_query_with_llm(
            last_prompt,
            (
                "Extract and return ONLY the item/entity name requested by the user. "
                "No quotes, no extra words, no commentary."
            ),
        )
    extracted_query = _clean_entity_lookup_query(extracted_query or last_prompt)

    with _timed_span(trace_id, "backend.entity_lookup", dataset=dataset, query=extracted_query):
        try:
            request_params = {
                "q": extracted_query,
                "limit": 10,
                "trace_id": trace_id,
            }
            target_url = f"{RECOMMENDER_BASE_URL}{RECOMMENDER_API_V1_PREFIX}/datasets/{dataset}/items/search"
            log.info("[ENTITY_LOOKUP] trace_id=%r GET %s params=%s", trace_id, target_url, request_params)
            remote_start = time.perf_counter()
            resp = http_requests.get(
                target_url,
                params=request_params,
                timeout=45,
                headers={"X-Trace-Id": trace_id},
            )
            remote_ms = round((time.perf_counter() - remote_start) * 1000, 2)
            resp.raise_for_status()

            data = resp.json()
            results = data.get("results", [])
            preview = [str(item.get("title", "")) for item in results[:5]]
            log.info(
                "[ENTITY_LOOKUP] trace_id=%r status=%s remote_ms=%s n_results=%s preview_titles=%s",
                trace_id,
                resp.status_code,
                remote_ms,
                len(results),
                preview,
            )

            if not results:
                return {
                    "trace_id": trace_id,
                    "messages": [
                        AIMessage(
                            content=f"No items found for '{extracted_query}' in dataset `{dataset}`."
                        )
                    ],
                }

            content = (
                f"I found {len(results)} matching items for '{extracted_query}' in `{dataset}`. "
                "The best matches are shown below as cards."
            )
            return {
                "trace_id": trace_id,
                "result": {
                    "kind": "search_results",
                    "title": f"Search results for {extracted_query}",
                    "subtitle": f"Dataset: {dataset}",
                    "items": results,
                    "dataset": dataset,
                    "query": extracted_query,
                    "trace_id": trace_id,
                    "latency": _response_latency_payload(remote_ms, data),
                },
                "messages": [AIMessage(content=content)],
            }

        except http_requests.exceptions.ConnectionError:
            return {
                "trace_id": trace_id,
                "messages": [
                    AIMessage(content="Recommender microservice unavailable (http://localhost:8001).")
                ],
            }
        except Exception as exc:
            return {"trace_id": trace_id, "messages": [AIMessage(content=f"Lookup error: {exc}")]}

def item_similarity_node(state: AgentState, config: RunnableConfig) -> dict:
    """Find similar items through /mf/similar endpoint."""
    import requests as http_requests

    trace_id = _resolve_trace_id(state, config)
    dataset = _resolve_dataset(config)
    rec_model = _resolve_rec_model(config)
    last_prompt = _content_to_text(state["messages"][-1].content) if state.get("messages") else ""
    attrs = state.get("attributes", {})

    extracted_query = _safe_item_query_from_attributes(attrs, "")
    if not extracted_query:
        extracted_query = _extract_item_query_with_llm(
            last_prompt,
            (
                "Extract and return ONLY the reference item/entity for similarity search. "
                "No quotes, no extra words, no commentary."
            ),
        )

    similar_endpoint_map = {
        model_key: f"{RECOMMENDER_API_V1_PREFIX}/recommenders/{model_slug}/similar-items"
        for model_key, model_slug in MODEL_SLUG_MAP.items()
    }
    endpoint = similar_endpoint_map.get(
        rec_model,
        f"{RECOMMENDER_API_V1_PREFIX}/recommenders/matrix-factorization/similar-items",
    )

    with _timed_span(
        trace_id,
        "backend.item_similarity",
        dataset=dataset,
        model=rec_model,
        item_query=extracted_query,
    ):
        try:
            request_payload = {
                "item_title": extracted_query,
                "dataset": dataset,
                "top_k": 10,
                "trace_id": trace_id,
            }
            log.info("[ITEM_SIMILARITY] trace_id=%r POST %s%s payload=%s", trace_id, RECOMMENDER_BASE_URL, endpoint, request_payload)

            remote_start = time.perf_counter()
            resp = http_requests.post(
                f"{RECOMMENDER_BASE_URL}{endpoint}",
                json=request_payload,
                timeout=45,
                headers={"X-Trace-Id": trace_id},
            )
            remote_ms = round((time.perf_counter() - remote_start) * 1000, 2)
            resp.raise_for_status()

            data = resp.json()
            seed_title = data.get("seed_title", extracted_query)
            items = data.get("items", [])
            preview = [str(item.get("title", "")) for item in items[:5]]
            log.info(
                "[ITEM_SIMILARITY] trace_id=%r status=%s remote_ms=%s seed_title=%r n_items=%s preview_titles=%s",
                trace_id,
                resp.status_code,
                remote_ms,
                seed_title,
                len(items),
                preview,
            )

            if not items:
                return {"trace_id": trace_id, "messages": [AIMessage(content=f"No similar items found for '{seed_title}'.")]}

            content = (
                f"I found {len(items)} items similar to '{seed_title}' in `{dataset}` using `{data.get('model', rec_model)}`. "
                "The ranked list is rendered below."
            )
            return {
                "trace_id": trace_id,
                "result": {
                    "kind": "similar_items",
                    "title": f"Because you searched for {seed_title}",
                    "subtitle": f"Dataset: {dataset} | Model: {data.get('model', rec_model)}",
                    "items": items,
                    "dataset": dataset,
                    "rec_model": data.get("model", rec_model),
                    "seed_title": seed_title,
                    "trace_id": trace_id,
                    "latency": _response_latency_payload(remote_ms, data),
                },
                "messages": [AIMessage(content=content)],
            }

        except http_requests.exceptions.ConnectionError:
            return {
                "trace_id": trace_id,
                "messages": [
                    AIMessage(content="Recommender microservice unavailable (http://localhost:8001).")
                ],
            }
        except http_requests.exceptions.HTTPError as exc:
            detail = ""
            if exc.response is not None:
                try:
                    payload = exc.response.json()
                    detail = str(payload.get("detail", "")).strip()
                except Exception:
                    detail = exc.response.text.strip()

            if exc.response is not None and exc.response.status_code == 404 and detail.lower().startswith("run not trained"):
                return {"trace_id": trace_id, "messages": [AIMessage(content=detail)]}

            if exc.response is not None and exc.response.status_code == 404:
                return {
                    "trace_id": trace_id,
                    "messages": [
                        AIMessage(
                            content=f"Item '{extracted_query}' was not found in dataset `{dataset}`. "
                            "Please provide a more exact title."
                        )
                    ],
                }
            return {"trace_id": trace_id, "messages": [AIMessage(content=f"Similarity lookup error: {exc}")]}
        except Exception as exc:
            return {"trace_id": trace_id, "messages": [AIMessage(content=f"Similarity lookup error: {exc}")]}

def general_qa_node(state: AgentState, config: RunnableConfig) -> dict:
    """General QA node with optional weather tool usage."""
    trace_id = _resolve_trace_id(state, config)
    messages = state["messages"]
    last_message = messages[-1]
    last_content = _content_to_text(getattr(last_message, "content", ""))
    was_tool = messages and getattr(messages[-1], "type", "") == "tool"
    attributes = state.get("attributes", {})
    needs_weather_tool = bool(attributes.get("needs_weather_tool", False))

    with _timed_span(
        trace_id,
        "backend.general_qa",
        was_tool=was_tool,
        needs_weather_tool=needs_weather_tool,
        last_msg_type=getattr(last_message, "type", "?"),
    ):
        greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
        if getattr(last_message, "type", "") == "human" and last_content.lower().strip() in greetings:
            return {"trace_id": trace_id, "messages": [AIMessage(content="Hello! How can I help you today?")]}

        if (
            getattr(last_message, "type", "") == "human"
            and not was_tool
            and not needs_weather_tool
            and looks_like_project_context_question(last_content)
        ):
            try:
                mcp_start = time.perf_counter()
                answer = fetch_project_context_sync(last_content)
                mcp_ms = round((time.perf_counter() - mcp_start) * 1000, 2)
                log.info("[GENERAL_QA] trace_id=%r answered via benchmark_mcp in %sms", trace_id, mcp_ms)
                if answer:
                    return {"trace_id": trace_id, "messages": [AIMessage(content=answer)]}
            except Exception as exc:
                log.warning("[GENERAL_QA] trace_id=%r benchmark_mcp fallback failed: %s", trace_id, exc)

        try:
            deterministic_follow_up = None
            if getattr(last_message, "type", "") == "human" and not was_tool and not needs_weather_tool:
                deterministic_follow_up = _build_deterministic_recommendation_follow_up_answer(messages, last_content)
            if deterministic_follow_up:
                return {"trace_id": trace_id, "messages": [AIMessage(content=deterministic_follow_up)]}

            if was_tool:
                llm = _build_backend_chat_llm(temperature=0)
                tool_content = _content_to_text(messages[-1].content)
                system_prompt = SystemMessage(
                    content=(
                        "You are a helpful assistant. You just received a tool result. "
                        "Summarize it clearly and briefly in English."
                    )
                )
                messages_for_llm = [system_prompt] + messages
                messages_for_llm.append(
                    HumanMessage(content=f"Tool result: '{tool_content}'. Provide a concise user-facing answer.")
                )
            else:
                llm = _build_backend_chat_llm(temperature=0, tools=[get_weather] if needs_weather_tool else None)
                if needs_weather_tool:
                    log.info("[GENERAL_QA] trace_id=%r enabling weather tool via provider=%r", trace_id, _backend_llm_provider())
                system_prompt = SystemMessage(
                    content=(
                        "You are a recommender assistant.\n"
                        "Rules:\n"
                        "1. Use the weather tool only when weather information is explicitly requested.\n"
                        "2. For all other requests, answer directly in plain text.\n"
                        "3. Do not invent tool outputs.\n"
                        "4. Do not output JSON unless the user explicitly asks for JSON."
                    )
                )
                messages_for_llm = [system_prompt] + messages

            llm_start = time.perf_counter()
            response = llm.invoke(messages_for_llm)
            llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)
            log.info("[GENERAL_QA] trace_id=%r llm_invoke_ms=%s", trace_id, llm_ms)
            return {"trace_id": trace_id, "messages": [response]}
        except Exception as exc:
            content = (
                "I couldn't use the conversational LLM runtime for this message right now. "
                "Recommendation, search, similarity, and project-capability questions are still available.\n\n"
                f"Original request: {last_content}\n"
                f"Technical detail: {exc}"
            )
            return {"trace_id": trace_id, "messages": [AIMessage(content=content)]}

def route_intent(
    state: AgentState,
) -> Literal["user_recommendation", "entity_lookup", "item_similarity", "general_qa"]:
    """Route to the node selected by classifier."""
    intent = state.get("intent", "general_qa")
    trace_id = state.get("trace_id", "trace-missing")
    log.info("[ROUTE] trace_id=%r intent=%r", trace_id, intent)
    return intent


workflow = StateGraph(AgentState)

workflow.add_node("classifier", classifier_node)
workflow.add_node("user_recommendation", user_recommendation_node)
workflow.add_node("entity_lookup", entity_lookup_node)
workflow.add_node("item_similarity", item_similarity_node)
workflow.add_node("general_qa", general_qa_node)
workflow.add_node("tools", ToolNode([get_weather]))

workflow.add_edge(START, "classifier")
workflow.add_conditional_edges("classifier", route_intent)

workflow.add_edge("user_recommendation", END)
workflow.add_edge("entity_lookup", END)
workflow.add_edge("item_similarity", END)

workflow.add_conditional_edges("general_qa", tools_condition)
workflow.add_edge("tools", "general_qa")

graph = workflow.compile()

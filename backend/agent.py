import json
import logging
import os
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
    preference_statement_prefixes = (
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

    if any(marker in normalized for marker in recommendation_markers):
        return (
            "user_recommendation",
            {"item_query": "", "needs_weather_tool": False, "reason": "heuristic_recommendation"},
        )

    if normalized.startswith(preference_statement_prefixes):
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
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    openai_client = OpenAI(base_url=base_url, api_key=api_key)
    return instructor.from_openai(openai_client, mode=instructor.Mode.JSON)


def _classify_intent_with_instructor(user_text: str) -> IntentClassification:
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
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

    try:
        parsed = IntentClassification.model_validate(payload)
    except Exception:
        return default.intent, default.attributes.model_dump()
    return parsed.intent, parsed.attributes.model_dump()


def _safe_item_query_from_attributes(attrs: dict[str, Any], fallback_prompt: str) -> str:
    candidate = attrs.get("item_query")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return fallback_prompt.strip()


def _extract_item_query_with_llm(user_prompt: str, instruction: str) -> str:
    llm = ChatOllama(model="llama3.2", temperature=0)
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

        heuristic = _classify_intent_fast_path(user_text)
        if heuristic is not None:
            intent, attributes = heuristic
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
            log.info(
                "[CLASSIFIER] trace_id=%r via=instructor intent=%r attributes=%s",
                trace_id,
                intent,
                attributes,
            )
            return {"trace_id": trace_id, "intent": intent, "attributes": attributes}
        except Exception as instructor_exc:
            log.warning(
                "[CLASSIFIER] trace_id=%r instructor classification failed, fallback to Ollama parsing: %s",
                trace_id,
                instructor_exc,
            )

            llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)
            prompt = [
                SystemMessage(
                    content=(
                        "You are a workflow router for a recommender chatbot. "
                        "Classify the latest user message and extract attributes. "
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

            try:
                raw = llm.invoke(prompt).content
                parsed_dict = _parse_json_from_model_output(raw)
                intent, attributes = _validate_classifier_payload(parsed_dict)
                log.info(
                    "[CLASSIFIER] trace_id=%r via=ollama intent=%r attributes=%s",
                    trace_id,
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
    last_prompt = _content_to_text(state["messages"][-1].content) if state.get("messages") else ""
    effective_prompt = (
        f"{refinement_context}\n\nCurrent refinement request:\n{last_prompt}"
        if refinement_context
        else last_prompt
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
            resp = http_requests.post(url, json=payload_dict, timeout=recommendation_timeout)
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

            items_list = "\n".join(
                f"{i + 1}. {item['title']}" + (f" ({item['genres']})" if item.get("genres") else "")
                for i, item in enumerate(results)
            )
            content = (
                f"I found {len(results)} matching items for '{extracted_query}' in `{dataset}`. "
                "The best matches are shown below as cards.\n\n"
                f"Quick list:\n{items_list}"
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

        if was_tool:
            llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)
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
            llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)
            if needs_weather_tool:
                log.info("[GENERAL_QA] trace_id=%r enabling weather tool", trace_id)
                llm = llm.bind_tools([get_weather])
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

        try:
            llm_start = time.perf_counter()
            response = llm.invoke(messages_for_llm)
            llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)
            log.info("[GENERAL_QA] trace_id=%r llm_invoke_ms=%s", trace_id, llm_ms)
            return {"trace_id": trace_id, "messages": [response]}
        except Exception as exc:
            content = (
                f"[MOCK - INTENT: general_qa]\n"
                f"General conversation. User said: '{last_content}'\n\n"
                f"(Ollama may be unavailable. Error: {exc})"
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

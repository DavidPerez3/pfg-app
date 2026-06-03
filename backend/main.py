from __future__ import annotations

from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent import graph
from config import settings
from hitl_utils import build_hitl_refinement_context, is_recommendation_refinement
from memory_manager import memory_manager
from shared.contracts import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DatasetUsersResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackSummary,
    MemoryResponse,
    ShortTermMemoryResponse,
)
from state_store import store


app = FastAPI(title="PFG App Backend", version="0.3.0")
API_V1_PREFIX = "/api/v1"

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        role = message.get("role") or message.get("type")
        return str(role or "assistant")

    msg_type = getattr(message, "type", "")
    mapping = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }
    return mapping.get(msg_type, "assistant")


def _message_to_contract(message: Any) -> ChatMessage:
    if isinstance(message, dict):
        return ChatMessage(
            role=str(message.get("role") or message.get("type") or "assistant"),
            content=_content_to_text(message.get("content", "")),
        )
    return ChatMessage(
        role=_message_role(message),
        content=_content_to_text(getattr(message, "content", "")),
    )


def _check_http_health(url: str, *, optional: bool = False) -> str:
    try:
        response = requests.get(url, timeout=3)
        if response.ok:
            return "ok"
        return f"http {response.status_code}"
    except Exception as exc:
        if optional:
            return f"optional-unavailable: {exc}"
        return f"error: {exc}"


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "service": "pfg-app-backend",
        "version": "0.3.0",
        "defaults": {
            "dataset": settings.default_dataset,
            "rec_model": settings.default_rec_model,
        },
    }


@app.get(f"{API_V1_PREFIX}/health")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "pfg-app-backend",
        "endpoints": [
            f"{API_V1_PREFIX}/threads/{{thread_id}}/messages",
            f"{API_V1_PREFIX}/threads/{{thread_id}}/feedback",
            f"{API_V1_PREFIX}/datasets/{{dataset}}/users",
            f"{API_V1_PREFIX}/feedback/summary",
            f"{API_V1_PREFIX}/users/{{user_id}}/memory",
            f"{API_V1_PREFIX}/threads/{{thread_id}}/memory",
            f"{API_V1_PREFIX}/threads/{{thread_id}}",
            f"{API_V1_PREFIX}/health/detailed",
            "/chat",
            "/dataset-users",
            "/feedback",
            "/feedback/summary",
            "/memory/{user_id}",
            "/memory/long-term/{user_id}",
            "/memory/short-term/{thread_id}",
            "/threads/{thread_id}",
            "/session/{thread_id}",
            "/health/detailed",
        ],
    }


@app.get(f"{API_V1_PREFIX}/health/detailed")
@app.get("/health/detailed")
def health_detailed():
    checks = {
        "app_state_store": store.db_health()["status"],
        "short_term_memory": memory_manager.short_term_health()["status"],
        "long_term_memory": memory_manager.long_term_health()["status"],
        "recommender": _check_http_health(f"{settings.recommender_base_url}{API_V1_PREFIX}/health"),
        "ollama": _check_http_health(f"{settings.ollama_base_url}/api/tags", optional=True),
    }
    mandatory = [checks["app_state_store"], checks["recommender"]]
    overall = "ok" if all(value == "ok" for value in mandatory) else "degraded"
    return {
        "status": overall,
        "checks": checks,
        "defaults": {
            "dataset": settings.default_dataset,
            "rec_model": settings.default_rec_model,
        },
    }


async def _chat_impl(request: ChatRequest, *, thread_id_override: str | None = None):
    try:
        thread_id = thread_id_override or request.thread_id or "default-thread"
        user_id = request.user_id or "anonymous"
        latest_user_message = request.messages[-1].content if request.messages else ""
        input_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        session_history = memory_manager.get_session(thread_id)
        relevant_memory = memory_manager.retrieve_relevant(user_id, latest_user_message, top_k=5)
        latest_feedback = store.get_latest_feedback(thread_id)
        latest_recommendation_event = store.get_latest_recommendation_event(thread_id)
        hitl_refinement_context = None

        if is_recommendation_refinement(latest_user_message) and latest_recommendation_event:
            hitl_refinement_context = build_hitl_refinement_context(
                latest_user_prompt=str(latest_recommendation_event.get("user_message") or ""),
                latest_assistant_text=str(
                    latest_recommendation_event.get("assistant_message") or ""
                ),
                latest_feedback=latest_feedback,
            )

        if len(input_messages) <= 1 and session_history:
            input_messages = [
                {"role": message.role, "content": message.content}
                for message in session_history
            ] + input_messages

        if relevant_memory:
            memory_prompt = (
                "Persistent user memory facts. Use them only if relevant to the current request:\n"
                + "\n".join(f"- {fact}" for fact in relevant_memory)
            )
            input_messages = [{"role": "system", "content": memory_prompt}] + input_messages

        if hitl_refinement_context:
            input_messages = [
                {
                    "role": "system",
                    "content": (
                        "Human-in-the-loop refinement context for this thread:\n"
                        + hitl_refinement_context
                    ),
                }
            ] + input_messages

        selected_dataset = request.dataset or settings.default_dataset
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "trace_id": request.trace_id,
                "dataset": selected_dataset,
                "rec_model": request.rec_model,
                "dataset_user_id": request.dataset_user_id,
                "hitl_refinement_active": bool(hitl_refinement_context),
                "hitl_refinement_context": hitl_refinement_context,
            }
        }

        selected_model = request.rec_model or settings.default_rec_model

        result = graph.invoke({"messages": input_messages}, config=config)
        response_messages = [_message_to_contract(msg) for msg in result.get("messages", [])]

        if request.messages and response_messages:
            store.record_conversation_event(
                thread_id=thread_id,
                user_id=user_id,
                trace_id=result.get("trace_id") or request.trace_id,
                intent=result.get("intent"),
                dataset=selected_dataset,
                rec_model=selected_model,
                user_message=request.messages[-1].content,
                assistant_message=response_messages[-1].content,
            )

        new_facts = memory_manager.extract_candidate_facts(latest_user_message)
        if new_facts:
            memory_manager.store_facts(user_id, new_facts, source="chat")

        return ChatResponse(
            messages=response_messages,
            trace_id=result.get("trace_id") or request.trace_id,
            intent=result.get("intent"),
            dataset=selected_dataset,
            rec_model=result.get("result", {}).get("rec_model")
            if isinstance(result.get("result"), dict)
            else selected_model,
            result=result.get("result"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await _chat_impl(request)


@app.post(f"{API_V1_PREFIX}/threads/{{thread_id}}/messages", response_model=ChatResponse)
async def post_thread_message(thread_id: str, request: ChatRequest):
    return await _chat_impl(request, thread_id_override=thread_id)


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    feedback_id = store.store_feedback(request)
    return FeedbackResponse(status="stored", feedback_id=feedback_id)


@app.post(f"{API_V1_PREFIX}/threads/{{thread_id}}/feedback", response_model=FeedbackResponse)
async def submit_thread_feedback(thread_id: str, request: FeedbackRequest):
    merged = request.model_copy(update={"thread_id": thread_id})
    feedback_id = store.store_feedback(merged)
    return FeedbackResponse(status="stored", feedback_id=feedback_id)


@app.get("/dataset-users", response_model=DatasetUsersResponse)
async def dataset_users(dataset: str | None = None, limit: int = 25):
    target_dataset = dataset or settings.default_dataset
    try:
        response = requests.get(
            f"{settings.recommender_base_url}{API_V1_PREFIX}/datasets/{target_dataset}/users",
            params={"limit": limit},
            timeout=10,
        )
        response.raise_for_status()
        return DatasetUsersResponse.model_validate(response.json())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(f"{API_V1_PREFIX}/datasets/{{dataset}}/users", response_model=DatasetUsersResponse)
async def dataset_users_v1(dataset: str, limit: int = 25):
    return await dataset_users(dataset=dataset, limit=limit)


@app.get(f"{API_V1_PREFIX}/feedback/summary", response_model=FeedbackSummary)
@app.get("/feedback/summary", response_model=FeedbackSummary)
async def feedback_summary():
    return store.feedback_summary()


@app.get(f"{API_V1_PREFIX}/users/{{user_id}}/memory", response_model=MemoryResponse)
@app.get("/memory/{user_id}", response_model=MemoryResponse)
async def get_memory(user_id: str):
    facts = memory_manager.get_all_facts(user_id)
    return MemoryResponse(user_id=user_id, facts=facts, count=len(facts))


@app.get(f"{API_V1_PREFIX}/users/{{user_id}}/memory/long-term", response_model=MemoryResponse)
@app.get("/memory/long-term/{user_id}", response_model=MemoryResponse)
async def get_long_term_memory(user_id: str):
    facts = memory_manager.get_all_facts(user_id)
    return MemoryResponse(user_id=user_id, facts=facts, count=len(facts))


@app.get(f"{API_V1_PREFIX}/threads/{{thread_id}}/memory", response_model=ShortTermMemoryResponse)
@app.get("/memory/short-term/{thread_id}", response_model=ShortTermMemoryResponse)
async def get_short_term_memory(thread_id: str):
    messages = memory_manager.get_session(thread_id)
    return ShortTermMemoryResponse(
        thread_id=thread_id,
        messages=messages,
        count=len(messages),
        window_hours=settings.short_term_memory_hours,
    )


@app.delete(f"{API_V1_PREFIX}/users/{{user_id}}/memory")
@app.delete("/memory/{user_id}")
async def delete_memory(user_id: str):
    memory_manager.delete_user_memory(user_id)
    return {"status": "deleted", "user_id": user_id}


@app.delete(f"{API_V1_PREFIX}/users/{{user_id}}/memory/long-term")
@app.delete("/memory/long-term/{user_id}")
async def delete_long_term_memory(user_id: str):
    memory_manager.delete_user_memory(user_id)
    return {"status": "deleted", "user_id": user_id}


@app.delete(f"{API_V1_PREFIX}/threads/{{thread_id}}/memory")
@app.delete("/memory/short-term/{thread_id}")
async def delete_short_term_memory(thread_id: str):
    memory_manager.clear_session(thread_id)
    return {"status": "cleared", "thread_id": thread_id}


@app.delete("/session/{thread_id}")
async def clear_session(thread_id: str):
    memory_manager.clear_session(thread_id)
    return {"status": "cleared", "thread_id": thread_id}


@app.delete(f"{API_V1_PREFIX}/threads/{{thread_id}}")
@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete one application thread worth of persisted conversation events.

    The older `/session/{thread_id}` route is kept as a compatibility alias, but
    `/threads/{thread_id}` is the clearer public name for the current product UI.
    """
    store.delete_thread(thread_id)
    return {"status": "deleted", "thread_id": thread_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

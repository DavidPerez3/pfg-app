from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import timedelta
from threading import Thread
from typing import Any

import requests
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from config import settings


log = logging.getLogger(__name__)

PROJECT_CONTEXT_PATTERNS = (
    r"\bwhat datasets\b",
    r"\bwhich datasets\b",
    r"\bsupported datasets\b",
    r"\bwhat models\b",
    r"\bwhich models\b",
    r"\brecommendation models\b",
    r"\bavailable models\b",
    r"\bsupported models\b",
    r"\bhow (?:is|does) (?:this project|the system|the app)\b",
    r"\bdeployment\b",
    r"\barchitecture\b",
    r"\bmemory\b",
    r"\bfeedback\b",
    r"\bmcp\b",
    r"\bmodel context protocol\b",
    r"\bgemini\b",
    r"\bollama\b",
    r"\bwhat tools\b",
    r"\bwhat services\b",
    r"\bdifference between backend and recommender\b",
    r"\bbackend and recommender\b",
)


def looks_like_project_context_question(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in PROJECT_CONTEXT_PATTERNS)


def _extract_answer_from_json_text(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if isinstance(payload, dict):
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    return None


def _extract_text_from_mcp_result(result: Any) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parsed_answer = _extract_answer_from_json_text(text.strip())
                if parsed_answer:
                    return parsed_answer
                texts.append(text.strip())
        if texts:
            return "\n".join(texts)

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        answer = structured.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        return json.dumps(structured, ensure_ascii=False, indent=2)

    if hasattr(result, "model_dump"):
        dumped = result.model_dump()
        if isinstance(dumped, dict):
            structured = dumped.get("structuredContent")
            if isinstance(structured, dict):
                answer = structured.get("answer")
                if isinstance(answer, str) and answer.strip():
                    return answer.strip()
            dumped_content = dumped.get("content")
            if isinstance(dumped_content, list):
                for item in dumped_content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parsed_answer = _extract_answer_from_json_text(text.strip())
                        if parsed_answer:
                            return parsed_answer
            return json.dumps(dumped, ensure_ascii=False, indent=2)

    return str(result)


async def fetch_project_context(question: str) -> str:
    async with streamablehttp_client(settings.benchmark_mcp_url) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "answer_project_question",
                arguments={"question": question},
                read_timeout_seconds=timedelta(seconds=15),
            )
            return _extract_text_from_mcp_result(result)


def _run_async_in_thread(coro: Any) -> Any:
    state: dict[str, Any] = {}

    def runner() -> None:
        try:
            state["result"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - defensive threading wrapper
            state["error"] = exc

    thread = Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in state:
        raise state["error"]
    return state.get("result")


def fetch_project_context_sync(question: str) -> str:
    return str(_run_async_in_thread(fetch_project_context(question)) or "").strip()


def project_mcp_health() -> str:
    health_url = settings.benchmark_mcp_url.removesuffix("/mcp") + "/health"
    try:
        response = requests.get(health_url, timeout=3)
        if response.ok:
            return "ok"
        return f"http {response.status_code}"
    except Exception as exc:
        return f"optional-unavailable: {exc}"

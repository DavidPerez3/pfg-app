from __future__ import annotations

import sys
import time

import requests


BACKEND = "http://localhost:8000"
RECOMMENDER = "http://localhost:8001"
ELASTICSEARCH = "http://localhost:9200"
BENCHMARK_MCP = "http://localhost:8010"


def wait_ok(url: str, timeout_seconds: int = 90) -> None:
    deadline = time.time() + timeout_seconds
    last_error = "unknown error"
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                return
            last_error = f"http {response.status_code}: {response.text[:200]}"
        except Exception as exc:  # pragma: no cover - defensive
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    wait_ok(f"{ELASTICSEARCH}/")
    wait_ok(f"{BENCHMARK_MCP}/health")
    wait_ok(f"{RECOMMENDER}/api/v1/health")
    wait_ok(f"{BACKEND}/api/v1/health")

    backend_health = requests.get(f"{BACKEND}/api/v1/health", timeout=10).json()
    require(backend_health["status"] == "ok", "Backend /api/v1/health is not ok")

    detailed = requests.get(f"{BACKEND}/health/detailed", timeout=10).json()
    require(detailed["checks"]["app_state_store"] == "ok", "App state store is not healthy")
    require(detailed["checks"]["recommender"] == "ok", "Recommender is not healthy from backend view")
    require(detailed["checks"]["benchmark_mcp"] == "ok", "Benchmark MCP is not healthy from backend view")

    mcp_capabilities = requests.get(f"{BENCHMARK_MCP}/capabilities", timeout=10).json()
    require("answer_project_question" in mcp_capabilities["tools"], "Benchmark MCP tool list is incomplete")

    users = requests.get(
        f"{RECOMMENDER}/api/v1/datasets/movielens/users",
        params={"limit": 3},
        timeout=15,
    ).json()
    require(users["dataset"] == "movielens", "Dataset users endpoint returned wrong dataset")
    require(len(users["users"]) >= 1, "Dataset users endpoint returned no users")

    search = requests.get(
        f"{RECOMMENDER}/api/v1/datasets/movielens/items/search",
        params={"q": "matrix", "limit": 3},
        timeout=20,
    ).json()
    titles = [item["title"] for item in search["results"]]
    require(any("matrix" in title.lower() for title in titles), "Recommender search did not return Matrix")

    chat_payload = {
        "messages": [{"role": "user", "content": "find matrix"}],
        "thread_id": "smoke-thread",
        "user_id": "smoke@example.com",
        "dataset": "movielens",
        "rec_model": "matrix_factorization",
    }
    chat = requests.post(f"{BACKEND}/chat", json=chat_payload, timeout=30).json()
    require(chat["intent"] == "entity_lookup", "Backend chat did not route to entity_lookup")
    require(chat["result"]["kind"] == "search_results", "Backend chat did not return search_results")
    chat_titles = [item["title"] for item in chat["result"]["items"]]
    require(any("matrix" in title.lower() for title in chat_titles), "Backend chat lookup did not return Matrix")

    feedback = requests.post(
        f"{BACKEND}/feedback",
        json={
            "user_id": "smoke@example.com",
            "thread_id": "smoke-thread",
            "rating": 5,
            "comment": "smoke ok",
            "message_index": 1,
        },
        timeout=15,
    ).json()
    require(feedback["status"] == "stored", "Feedback endpoint did not store feedback")

    summary = requests.get(f"{BACKEND}/feedback/summary", timeout=15).json()
    require(summary["count"] >= 1, "Feedback summary count did not increase as expected")

    print("Smoke stack checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[SMOKE][FAIL] {exc}", file=sys.stderr)
        raise

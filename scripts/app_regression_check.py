from __future__ import annotations

import json
import time

import requests


BASE_URL = "http://127.0.0.1:8000"
CHAT_URL = f"{BASE_URL}/chat"


def send(thread_id: str, user_id: str, prompt: str, *, dataset: str = "movielens", rec_model: str = "mf") -> dict:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "thread_id": thread_id,
        "user_id": user_id,
        "dataset": dataset,
        "rec_model": rec_model,
    }
    response = requests.post(CHAT_URL, json=body, timeout=120)
    response.raise_for_status()
    return response.json()


def main() -> None:
    results: list[dict] = []

    mcp_models = send("reg-mcp-models", "reg-mcp@example.com", "What recommendation models are available?")
    results.append(
        {
            "case": "mcp_models",
            "intent": mcp_models.get("intent"),
            "result_kind": (mcp_models.get("result") or {}).get("kind") if isinstance(mcp_models.get("result"), dict) else None,
            "message": mcp_models["messages"][-1]["content"],
        }
    )

    mcp_purpose = send("reg-mcp-purpose", "reg-mcp@example.com", "What is MCP used for in this project?")
    results.append(
        {
            "case": "mcp_purpose",
            "intent": mcp_purpose.get("intent"),
            "result_kind": (mcp_purpose.get("result") or {}).get("kind") if isinstance(mcp_purpose.get("result"), dict) else None,
            "message": mcp_purpose["messages"][-1]["content"],
        }
    )

    mcp_arch = send("reg-mcp-arch", "reg-mcp@example.com", "What is the difference between backend and recommender?")
    results.append(
        {
            "case": "mcp_backend_vs_recommender",
            "intent": mcp_arch.get("intent"),
            "result_kind": (mcp_arch.get("result") or {}).get("kind") if isinstance(mcp_arch.get("result"), dict) else None,
            "message": mcp_arch["messages"][-1]["content"],
        }
    )

    lookup = send("reg-lookup", "reg-lookup@example.com", "find matrix")
    results.append(
        {
            "case": "lookup_matrix",
            "intent": lookup.get("intent"),
            "result_kind": (lookup.get("result") or {}).get("kind") if isinstance(lookup.get("result"), dict) else None,
            "has_quick_list": "Quick list:" in lookup["messages"][-1]["content"],
            "message": lookup["messages"][-1]["content"],
        }
    )

    send("reg-followup", "reg-follow@example.com", "recommend me sci-fi movies")
    followup = send("reg-followup", "reg-follow@example.com", "Which one should I start with first?")
    results.append(
        {
            "case": "followup_pick_one",
            "intent": followup.get("intent"),
            "message": followup["messages"][-1]["content"],
        }
    )

    send("reg-pref-thread", "reg-pref@example.com", "I like sci-fi and thrillers")
    same_thread = send("reg-pref-thread", "reg-pref@example.com", "recommend me something")
    results.append(
        {
            "case": "same_thread_preference",
            "intent": same_thread.get("intent"),
            "top5": [item["title"] for item in (same_thread.get("result") or {}).get("items", [])[:5]],
        }
    )

    send("reg-pref-store", "reg-pref-cross@example.com", "I like sci-fi and thrillers")
    time.sleep(1)
    cross_thread = send("reg-pref-other-thread", "reg-pref-cross@example.com", "recommend me something")
    results.append(
        {
            "case": "cross_thread_preference",
            "intent": cross_thread.get("intent"),
            "top5": [item["title"] for item in (cross_thread.get("result") or {}).get("items", [])[:5]],
        }
    )

    health = requests.get(f"{BASE_URL}/health/detailed", timeout=30).json()
    results.append({"case": "health", "checks": health.get("checks", {})})

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

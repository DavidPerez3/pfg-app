from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_BACKEND = "http://localhost:8000"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "benchmark_results"


@dataclass(frozen=True)
class Scenario:
    name: str
    method: str
    path: str
    params: dict[str, Any] | None = None
    body_factory: callable | None = None
    timeout_seconds: int = 60


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _thread_payload(
    *,
    content: str,
    dataset: str,
    rec_model: str,
    dataset_user_id: str | None = None,
) -> dict[str, Any]:
    trace_id = f"bench-{rec_model}-{int(time.time() * 1000)}"
    return {
        "messages": [{"role": "user", "content": content}],
        "thread_id": trace_id,
        "user_id": "latency-benchmark@example.com",
        "trace_id": trace_id,
        "client_sent_at_ms": time.time() * 1000,
        "dataset": dataset,
        "rec_model": rec_model,
        "dataset_user_id": dataset_user_id,
    }


def default_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="health_detailed_backend",
            method="GET",
            path="/api/v1/health/detailed",
            timeout_seconds=10,
        ),
        Scenario(
            name="dataset_users_movielens_mf",
            method="GET",
            path="/api/v1/datasets/movielens/users",
            params={"limit": 25, "rec_model": "mf"},
            timeout_seconds=30,
        ),
        Scenario(
            name="chat_lookup_movielens",
            method="POST",
            path="/api/v1/threads/bench-lookup/messages",
            body_factory=lambda: _thread_payload(
                content="find matrix",
                dataset="movielens",
                rec_model="mf",
            ),
            timeout_seconds=45,
        ),
        Scenario(
            name="chat_recommend_mf_movielens",
            method="POST",
            path="/api/v1/threads/bench-mf/messages",
            body_factory=lambda: _thread_payload(
                content="recommend me movies",
                dataset="movielens",
                rec_model="mf",
            ),
            timeout_seconds=60,
        ),
        Scenario(
            name="chat_recommend_two_tower_movielens",
            method="POST",
            path="/api/v1/threads/bench-tt/messages",
            body_factory=lambda: _thread_payload(
                content="recommend me movies",
                dataset="movielens",
                rec_model="two_tower",
            ),
            timeout_seconds=60,
        ),
        Scenario(
            name="chat_recommend_ttwd_movielens",
            method="POST",
            path="/api/v1/threads/bench-ttwd/messages",
            body_factory=lambda: _thread_payload(
                content="recommend me movies",
                dataset="movielens",
                rec_model="two_tower_wide_deep",
            ),
            timeout_seconds=60,
        ),
        Scenario(
            name="chat_recommend_sasrec_movielens",
            method="POST",
            path="/api/v1/threads/bench-sasrec/messages",
            body_factory=lambda: _thread_payload(
                content="recommend me movies",
                dataset="movielens",
                rec_model="sasrec",
            ),
            timeout_seconds=60,
        ),
    ]


def optional_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="dataset_users_yelp_mf",
            method="GET",
            path="/api/v1/datasets/yelp/users",
            params={"limit": 25, "rec_model": "mf"},
            timeout_seconds=45,
        ),
        Scenario(
            name="chat_recommend_llm_rag_movielens",
            method="POST",
            path="/api/v1/threads/bench-rag/messages",
            body_factory=lambda: _thread_payload(
                content="recommend me movies",
                dataset="movielens",
                rec_model="llm_rag",
            ),
            timeout_seconds=120,
        ),
    ]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 2)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    value = ordered[low] * (1 - frac) + ordered[high] * frac
    return round(value, 2)


def summarize_metric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None, "p50": None, "p95": None}
    return {
        "mean": round(statistics.fmean(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
    }


def run_scenario(base_url: str, scenario: Scenario) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{scenario.path}"
    payload = scenario.body_factory() if scenario.body_factory else None
    started = time.perf_counter()
    result: dict[str, Any] = {
        "scenario": scenario.name,
        "method": scenario.method,
        "url": url,
        "started_at": _now_iso(),
    }
    try:
        if scenario.method == "GET":
            response = requests.get(url, params=scenario.params, timeout=scenario.timeout_seconds)
        elif scenario.method == "POST":
            response = requests.post(url, params=scenario.params, json=payload, timeout=scenario.timeout_seconds)
        else:
            raise ValueError(f"Unsupported method: {scenario.method}")
        roundtrip_ms = round((time.perf_counter() - started) * 1000, 2)
        result["http_status"] = response.status_code
        result["roundtrip_ms"] = roundtrip_ms
        try:
            data = response.json()
        except Exception:
            data = {"raw_text": response.text[:1000]}
        result["ok"] = bool(response.ok)
        result["response"] = data
        if isinstance(data, dict):
            result["detail"] = data.get("detail")
            result["latency"] = data.get("latency") or (
                data.get("result", {}).get("latency") if isinstance(data.get("result"), dict) else None
            )
            result["intent"] = data.get("intent")
            if isinstance(data.get("result"), dict):
                result["result_kind"] = data["result"].get("kind")
                result["result_title"] = data["result"].get("title")
        return result
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
        result["roundtrip_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(item["scenario"], []).append(item)

    summary: dict[str, Any] = {}
    for scenario_name, runs in grouped.items():
        roundtrips = [float(run["roundtrip_ms"]) for run in runs if isinstance(run.get("roundtrip_ms"), (int, float))]
        backend_totals = []
        backend_to_recommender = []
        recommender_totals = []
        statuses = []
        errors = []
        for run in runs:
            statuses.append(run.get("http_status"))
            if run.get("error"):
                errors.append(run["error"])
            latency = run.get("latency")
            if isinstance(latency, dict):
                for key, target in (
                    ("backend_total_ms", backend_totals),
                    ("backend_to_recommender_http_ms", backend_to_recommender),
                    ("recommender_total_ms", recommender_totals),
                ):
                    value = latency.get(key)
                    if isinstance(value, (int, float)):
                        target.append(float(value))
        summary[scenario_name] = {
            "runs": len(runs),
            "successes": sum(1 for run in runs if run.get("ok")),
            "failures": sum(1 for run in runs if not run.get("ok")),
            "http_statuses": statuses,
            "roundtrip_ms": summarize_metric(roundtrips),
            "backend_total_ms": summarize_metric(backend_totals),
            "backend_to_recommender_http_ms": summarize_metric(backend_to_recommender),
            "recommender_total_ms": summarize_metric(recommender_totals),
            "errors": errors,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end latency of the PFG app runtime.")
    parser.add_argument("--backend-base-url", default=DEFAULT_BACKEND)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    scenarios = default_scenarios()
    if args.include_optional:
        scenarios.extend(optional_scenarios())

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        for repeat_idx in range(args.repeats):
            print(f"[LATENCY][RUN] scenario={scenario.name} repeat={repeat_idx + 1}/{args.repeats}")
            run_result = run_scenario(args.backend_base_url, scenario)
            run_result["repeat"] = repeat_idx + 1
            results.append(run_result)

    summary = aggregate(results)
    report = {
        "generated_at": _now_iso(),
        "backend_base_url": args.backend_base_url,
        "repeats": args.repeats,
        "include_optional": args.include_optional,
        "summary": summary,
        "results": results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"latency_benchmark_{timestamp}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("[LATENCY][SUMMARY]")
    for scenario_name, item in summary.items():
        print(
            json.dumps(
                {
                    "scenario": scenario_name,
                    "successes": item["successes"],
                    "failures": item["failures"],
                    "roundtrip_ms": item["roundtrip_ms"],
                    "backend_total_ms": item["backend_total_ms"],
                    "backend_to_recommender_http_ms": item["backend_to_recommender_http_ms"],
                    "recommender_total_ms": item["recommender_total_ms"],
                }
            )
        )
    print()
    print(f"[LATENCY][REPORT] {json_path}")


if __name__ == "__main__":
    main()

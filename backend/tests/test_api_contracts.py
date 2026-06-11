import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os
import tempfile

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import _test_env  # noqa: F401
import main as backend_main


class BackendApiContractsTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(backend_main.app)

    def test_root_exposes_service_defaults(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "pfg-app-backend")
        self.assertIn("defaults", payload)
        self.assertIn("dataset", payload["defaults"])
        self.assertIn("rec_model", payload["defaults"])

    def test_health_v1_alias_is_available(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("/api/v1/threads/{thread_id}/messages", payload["endpoints"])

    @patch.object(backend_main, "store")
    @patch.object(backend_main, "memory_manager")
    @patch.object(backend_main, "_llm_runtime_health")
    @patch.object(backend_main, "project_mcp_health")
    @patch.object(backend_main, "_check_http_health")
    def test_health_detailed_degrades_when_recommender_is_down(
        self,
        mock_check_http_health,
        mock_project_mcp_health,
        mock_llm_runtime_health,
        mock_memory_manager,
        mock_store,
    ):
        mock_store.db_health.return_value = {"status": "ok"}
        mock_memory_manager.short_term_health.return_value = {"status": "ok"}
        mock_memory_manager.long_term_health.return_value = {"status": "ok"}
        mock_check_http_health.return_value = "error: boom"
        mock_project_mcp_health.return_value = "ok"
        mock_llm_runtime_health.return_value = "configured: gemini (gemini-2.5-flash-lite)"

        response = self.client.get("/health/detailed")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["checks"]["app_state_store"], "ok")
        self.assertTrue(payload["checks"]["recommender"].startswith("error:"))
        self.assertEqual(payload["checks"]["benchmark_mcp"], "ok")

    @patch.object(backend_main.requests, "get")
    def test_dataset_users_proxies_recommender_payload(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "dataset": "movielens",
            "users": [{"user_id": "42", "interaction_count": 99}],
            "total_available": 1,
            "latency": {"recommender_total_ms": 12.5},
        }
        mock_get.return_value = mock_response

        response = self.client.get("/dataset-users?dataset=movielens&limit=5&rec_model=mf")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["dataset"], "movielens")
        self.assertEqual(payload["users"][0]["user_id"], "42")
        self.assertEqual(payload["total_available"], 1)
        self.assertIn("latency", payload)
        self.assertGreaterEqual(payload["latency"]["backend_to_recommender_http_ms"], 0)
        request_kwargs = mock_get.call_args.kwargs
        self.assertEqual(request_kwargs["params"]["rec_model"], "mf")
        response_v1 = self.client.get("/api/v1/datasets/movielens/users?limit=5&rec_model=mf")
        self.assertEqual(response_v1.status_code, 200)

    @patch.object(backend_main, "store")
    @patch.object(backend_main, "memory_manager")
    @patch.object(backend_main.graph, "invoke")
    def test_chat_project_context_question_returns_general_qa_response(
        self,
        mock_invoke,
        mock_memory_manager,
        mock_store,
    ):
        mock_memory_manager.get_session.return_value = []
        mock_memory_manager.retrieve_relevant.return_value = []
        mock_memory_manager.extract_candidate_facts.return_value = []
        mock_store.get_latest_feedback.return_value = None
        mock_store.get_latest_recommendation_event.return_value = None
        mock_invoke.return_value = {
            "messages": [
                {"role": "user", "content": "What recommendation models are available?"},
                {"role": "assistant", "content": "Supported recommendation paradigms: Matrix Factorization, Two-Tower, Two-Tower + Wide&Deep, SASRec, and LLM + RAG."},
            ],
            "trace_id": "trace-mcp-route",
            "intent": "general_qa",
            "result": None,
        }

        response = self.client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "What recommendation models are available?"}],
                "thread_id": "thread-mcp-route",
                "user_id": "david@example.com",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["intent"], "general_qa")
        self.assertIn("Supported recommendation paradigms", payload["messages"][-1]["content"])

    @patch.object(backend_main, "store")
    @patch.object(backend_main, "memory_manager")
    @patch.object(backend_main.graph, "invoke")
    def test_chat_returns_structured_result_and_records_event(
        self,
        mock_invoke,
        mock_memory_manager,
        mock_store,
    ):
        mock_memory_manager.get_session.return_value = []
        mock_memory_manager.retrieve_relevant.return_value = ["I like sci-fi"]
        mock_memory_manager.extract_candidate_facts.return_value = ["I like sci-fi"]
        mock_memory_manager.store_facts.return_value = ["I like sci-fi"]
        mock_store.get_latest_feedback.return_value = None
        mock_store.get_latest_recommendation_event.return_value = None
        mock_invoke.return_value = {
            "messages": [
                {"role": "user", "content": "recommend me movies"},
                {"role": "assistant", "content": "Here are some ideas."},
            ],
            "trace_id": "trace-123",
            "intent": "user_recommendation",
            "result": {
                "kind": "recommendations",
                "title": "Recommendations",
                "dataset": "movielens",
                "rec_model": "mf",
                "items": [{"title": "Movie A", "score": 1.0, "genres": "Drama"}],
                "follow_up_prompts": ["Give me a more mainstream version of these recommendations"],
            },
        }

        response = self.client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "recommend me movies"}],
                "thread_id": "thread-1",
                "user_id": "david@example.com",
                "client_sent_at_ms": 0,
                "dataset": "movielens",
                "rec_model": "mf",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["trace_id"], "trace-123")
        self.assertEqual(payload["intent"], "user_recommendation")
        self.assertEqual(payload["result"]["kind"], "recommendations")
        self.assertEqual(payload["messages"][-1]["content"], "Here are some ideas.")
        self.assertEqual(len(payload["result"]["follow_up_prompts"]), 1)
        self.assertIn("latency", payload)
        self.assertGreaterEqual(payload["latency"]["backend_total_ms"], 0)
        mock_store.record_conversation_event.assert_called_once()
        recorded_kwargs = mock_store.record_conversation_event.call_args.kwargs
        self.assertIn("Top recommended items: Movie A.", recorded_kwargs["assistant_message"])
        mock_memory_manager.store_facts.assert_called_once()

    @patch.object(backend_main, "store")
    @patch.object(backend_main, "memory_manager")
    @patch.object(backend_main.graph, "invoke")
    def test_thread_message_v1_route_overrides_thread_id(
        self,
        mock_invoke,
        mock_memory_manager,
        mock_store,
    ):
        mock_memory_manager.get_session.return_value = []
        mock_memory_manager.retrieve_relevant.return_value = []
        mock_memory_manager.extract_candidate_facts.return_value = []
        mock_store.get_latest_feedback.return_value = None
        mock_store.get_latest_recommendation_event.return_value = None
        mock_invoke.return_value = {
            "messages": [
                {"role": "user", "content": "recommend me movies"},
                {"role": "assistant", "content": "Here are some ideas."},
            ],
            "trace_id": "trace-456",
            "intent": "user_recommendation",
            "result": None,
        }

        response = self.client.post(
            "/api/v1/threads/thread-v1/messages",
            json={
                "messages": [{"role": "user", "content": "recommend me movies"}],
                "thread_id": "ignored-thread",
                "user_id": "david@example.com",
            },
        )

        self.assertEqual(response.status_code, 200)
        mock_store.record_conversation_event.assert_called_once()
        recorded_kwargs = mock_store.record_conversation_event.call_args.kwargs
        self.assertEqual(recorded_kwargs["thread_id"], "thread-v1")

    @patch.object(backend_main, "store")
    @patch.object(backend_main, "memory_manager")
    @patch.object(backend_main.graph, "invoke")
    def test_chat_marks_refinement_context_for_follow_up_recommendation(
        self,
        mock_invoke,
        mock_memory_manager,
        mock_store,
    ):
        mock_memory_manager.get_session.return_value = []
        mock_memory_manager.retrieve_relevant.return_value = []
        mock_memory_manager.extract_candidate_facts.return_value = []
        mock_store.get_latest_feedback.return_value = {
            "rating": 2,
            "comment": "Marked as needing work.",
        }
        mock_store.get_latest_recommendation_event.return_value = {
            "user_message": "recommend me films",
            "assistant_message": "I found 10 recommendations in movielens using mf.",
        }
        mock_invoke.return_value = {
            "messages": [
                {"role": "user", "content": "more comedy please"},
                {"role": "assistant", "content": "Here are some lighter ideas."},
            ],
            "trace_id": "trace-hitl",
            "intent": "user_recommendation",
            "result": None,
        }

        response = self.client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "more comedy please"}],
                "thread_id": "thread-hitl",
                "user_id": "david@example.com",
            },
        )

        self.assertEqual(response.status_code, 200)
        invoke_config = mock_invoke.call_args.kwargs["config"]["configurable"]
        self.assertTrue(invoke_config["hitl_refinement_active"])
        self.assertIn("Previous recommendation request", invoke_config["hitl_refinement_context"])

    @patch.object(backend_main, "store")
    @patch.object(backend_main, "memory_manager")
    @patch.object(backend_main.graph, "invoke")
    def test_chat_injects_recommendation_context_for_natural_follow_up_question(
        self,
        mock_invoke,
        mock_memory_manager,
        mock_store,
    ):
        mock_memory_manager.get_session.return_value = []
        mock_memory_manager.retrieve_relevant.return_value = []
        mock_memory_manager.extract_candidate_facts.return_value = []
        mock_store.get_latest_feedback.return_value = {
            "rating": 4,
            "comment": "Good direction.",
        }
        mock_store.get_latest_recommendation_event.return_value = {
            "user_message": "recommend me sci-fi films",
            "assistant_message": "Top recommended items: Arrival; Ex Machina; Blade Runner.",
        }
        mock_invoke.return_value = {
            "messages": [
                {"role": "user", "content": "Which one should I start with first?"},
                {"role": "assistant", "content": "Start with Arrival if you want the most accessible entry point."},
            ],
            "trace_id": "trace-follow-up",
            "intent": "general_qa",
            "result": None,
        }

        response = self.client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Which one should I start with first?"}],
                "thread_id": "thread-follow-up",
                "user_id": "david@example.com",
            },
        )

        self.assertEqual(response.status_code, 200)
        invoke_args = mock_invoke.call_args.args[0]
        invoke_config = mock_invoke.call_args.kwargs["config"]["configurable"]
        system_messages = [
            message["content"]
            for message in invoke_args["messages"]
            if message.get("role") == "system"
        ]
        self.assertFalse(invoke_config["hitl_refinement_active"])
        self.assertTrue(
            any("refers to the previous recommendation list" in content for content in system_messages)
        )


if __name__ == "__main__":
    unittest.main()

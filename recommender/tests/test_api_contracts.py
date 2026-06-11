import unittest
from pathlib import Path
from unittest.mock import patch
import sys

from fastapi.testclient import TestClient

RECOMMENDER_ROOT = Path(__file__).resolve().parents[1]
if str(RECOMMENDER_ROOT) not in sys.path:
    sys.path.insert(0, str(RECOMMENDER_ROOT))

import main as recommender_main


class RecommenderApiContractsTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(recommender_main.app)

    def test_health_lists_main_online_endpoints(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("/search", payload["endpoints"])
        self.assertIn("/rag", payload["endpoints"])
        response_v1 = self.client.get("/api/v1/health")
        self.assertEqual(response_v1.status_code, 200)
        self.assertIn("/api/v1/recommenders/{model}/recommendations", response_v1.json()["endpoints"])

    @patch.object(recommender_main, "_dataset_user_options")
    def test_dataset_users_trims_options_to_requested_limit(self, mock_options):
        mock_options.return_value = (
            [
                recommender_main.DatasetUserOption(user_id="1", interaction_count=10),
                recommender_main.DatasetUserOption(user_id="2", interaction_count=8),
                recommender_main.DatasetUserOption(user_id="3", interaction_count=5),
            ],
            3,
        )

        response = self.client.get("/dataset-users?dataset=movielens&limit=2")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["dataset"], "movielens")
        self.assertEqual(len(payload["users"]), 2)
        self.assertEqual(payload["total_available"], 3)
        self.assertIn("latency", payload)
        self.assertGreaterEqual(payload["latency"]["recommender_total_ms"], 0)
        response_v1 = self.client.get("/api/v1/datasets/movielens/users?limit=2")
        self.assertEqual(response_v1.status_code, 200)

    @patch.object(recommender_main, "_mf_trained_dataset_user_ids")
    @patch.object(recommender_main, "_dataset_user_options")
    def test_dataset_users_filters_to_mf_servable_users_when_model_is_mf(
        self,
        mock_options,
        mock_trained_user_ids,
    ):
        mock_options.return_value = (
            [
                recommender_main.DatasetUserOption(user_id="20", interaction_count=20),
                recommender_main.DatasetUserOption(user_id="10", interaction_count=15),
                recommender_main.DatasetUserOption(user_id="30", interaction_count=12),
            ],
            3,
        )
        mock_trained_user_ids.return_value = frozenset({"10", "30"})

        response = self.client.get("/api/v1/datasets/movielens/users?limit=25&rec_model=mf")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_available"], 2)
        self.assertEqual([user["user_id"] for user in payload["users"]], ["10", "30"])

    @patch.object(recommender_main, "_search_items_from_parquet")
    @patch.object(recommender_main, "_search_items_from_elasticsearch")
    def test_search_falls_back_to_parquet_when_elasticsearch_fails(
        self,
        mock_search_es,
        mock_search_parquet,
    ):
        mock_search_es.side_effect = RuntimeError("es down")
        mock_search_parquet.return_value = [
            recommender_main.RecommendedItem(
                title="The Matrix",
                score=1.0,
                genres="Action|Sci-Fi",
            )
        ]

        response = self.client.post(
            "/search",
            json={"query": "matrix", "dataset": "movielens", "top_k": 5},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query"], "matrix")
        self.assertEqual(payload["results"][0]["title"], "The Matrix")
        self.assertIn("latency", payload)
        self.assertGreaterEqual(payload["latency"]["recommender_total_ms"], 0)
        mock_search_parquet.assert_called_once()
        response_v1 = self.client.get(
            "/api/v1/datasets/movielens/items/search?q=matrix&limit=5",
        )
        self.assertEqual(response_v1.status_code, 200)
        self.assertEqual(response_v1.json()["results"][0]["title"], "The Matrix")

    def test_search_rejects_unsupported_dataset(self):
        response = self.client.post(
            "/search",
            json={"query": "matrix", "dataset": "unknown_dataset", "top_k": 5},
        )
        self.assertEqual(response.status_code, 400)

    @patch.object(recommender_main, "mf_recommend")
    def test_versioned_recommendation_dispatches_by_model_slug(self, mock_mf_recommend):
        mock_mf_recommend.return_value = recommender_main.RecommendResponse(
            user_id="u1",
            dataset="movielens",
            model="matrix_factorization",
            cold_start=True,
            items=[],
            trace_id="trace-1",
        )
        response = self.client.post(
            "/api/v1/recommenders/matrix-factorization/recommendations",
            json={"user_id": "u1", "dataset": "movielens", "prompt": "recommend", "top_k": 10},
        )
        self.assertEqual(response.status_code, 200)
        mock_mf_recommend.assert_called_once()
        self.assertIn("latency", response.json())

    def test_mf_recommendation_returns_standard_run_not_trained_message(self):
        with patch.object(recommender_main, "MODELS_ROOT", Path("C:/definitely-missing-pfg-model-weights")):
            response = self.client.post(
                "/api/v1/recommenders/matrix-factorization/recommendations",
                json={"user_id": "u1", "dataset": "movielens", "prompt": "recommend", "top_k": 10},
            )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["detail"],
            "Run not trained: Matrix Factorization is not available for dataset 'movielens'.",
        )

    def test_mf_similarity_returns_standard_run_not_trained_message(self):
        with patch.object(recommender_main, "MODELS_ROOT", Path("C:/definitely-missing-pfg-model-weights")):
            response = self.client.post(
                "/api/v1/recommenders/matrix-factorization/similar-items",
                json={"item_title": "The Matrix", "dataset": "movielens", "top_k": 10},
            )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["detail"],
            "Run not trained: Matrix Factorization is not available for dataset 'movielens'.",
        )

    def test_versioned_model_health_rejects_unknown_model(self):
        response = self.client.get("/api/v1/recommenders/unknown-model/health")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()

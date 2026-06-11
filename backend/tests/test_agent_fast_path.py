import unittest
from pathlib import Path
from unittest.mock import patch
import sys

from langchain_core.messages import HumanMessage, SystemMessage

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import _test_env  # noqa: F401
from agent import (
    _build_deterministic_recommendation_follow_up_answer,
    _build_effective_recommendation_prompt,
    _classify_intent_fast_path,
    _clean_entity_lookup_query,
    _collect_recommendation_supporting_context,
    _sanitize_intent_classification,
    _validate_classifier_payload,
    classifier_node,
    general_qa_node,
)


class AgentFastPathTests(unittest.TestCase):
    def test_preference_statement_with_genres_becomes_recommendation(self):
        intent, attributes = _classify_intent_fast_path("I like sci-fi and thrillers")
        self.assertEqual(intent, "user_recommendation")
        self.assertEqual(attributes["reason"], "heuristic_theme_recommendation")

    def test_standalone_theme_phrase_becomes_recommendation(self):
        intent, attributes = _classify_intent_fast_path("sci-fi and thrillers")
        self.assertEqual(intent, "user_recommendation")
        self.assertEqual(attributes["reason"], "heuristic_theme_recommendation")

    def test_non_genre_preference_statement_stays_general_qa(self):
        intent, attributes = _classify_intent_fast_path("I like pizza")
        self.assertEqual(intent, "general_qa")
        self.assertEqual(attributes["reason"], "heuristic_preference_statement")

    def test_clean_entity_lookup_query_strips_lookup_verbs(self):
        self.assertEqual(_clean_entity_lookup_query("find teardrop"), "teardrop")
        self.assertEqual(_clean_entity_lookup_query("show me matrix"), "matrix")
        self.assertEqual(_clean_entity_lookup_query("look up sony"), "sony")

    def test_sanitize_theme_prompt_reclassifies_lookup_to_recommendation(self):
        intent, attributes = _sanitize_intent_classification(
            "show me sci-fi and thrillers",
            "entity_lookup",
            {"item_query": "show me sci-fi and thrillers", "needs_weather_tool": False, "reason": "llm_lookup"},
        )
        self.assertEqual(intent, "user_recommendation")
        self.assertEqual(attributes["item_query"], "")
        self.assertEqual(attributes["reason"], "sanitized_theme_recommendation")

    def test_sanitize_theme_prompt_reclassifies_similarity_to_recommendation(self):
        intent, attributes = _sanitize_intent_classification(
            "I like sci-fi and thrillers",
            "item_similarity",
            {"item_query": "sci-fi and thrillers", "needs_weather_tool": False, "reason": "llm_similarity"},
        )
        self.assertEqual(intent, "user_recommendation")
        self.assertEqual(attributes["item_query"], "")
        self.assertEqual(attributes["reason"], "sanitized_theme_recommendation")

    def test_sanitize_specific_lookup_keeps_entity_lookup_and_cleans_query(self):
        intent, attributes = _sanitize_intent_classification(
            "show me The Matrix",
            "entity_lookup",
            {"item_query": "show me The Matrix", "needs_weather_tool": False, "reason": "llm_lookup"},
        )
        self.assertEqual(intent, "entity_lookup")
        self.assertEqual(attributes["item_query"], "The Matrix")

    def test_generic_retry_prompt_inherits_feedback_note(self):
        effective = _build_effective_recommendation_prompt(
            "another try please",
            (
                "Previous recommendation request: recommend me films\n"
                "Stored feedback note: Too romantic and not enough comedy\n"
                "Treat the current message as a refinement of the previous recommendation."
            ),
        )
        self.assertIn("Current refinement request:", effective)
        self.assertIn("Too romantic and not enough comedy", effective)

    def test_validate_classifier_payload_normalizes_common_malformed_fields(self):
        intent, attributes = _validate_classifier_payload(
            {
                "intent": "Recommendation",
                "item_query": "   Blade Runner   ",
                "needs_weather_tool": "YES",
                "reason": 123,
            }
        )
        self.assertEqual(intent, "user_recommendation")
        self.assertEqual(attributes["item_query"], "Blade Runner")
        self.assertTrue(attributes["needs_weather_tool"])
        self.assertEqual(attributes["reason"], "123")

    def test_validate_classifier_payload_uses_nested_attributes_when_present(self):
        intent, attributes = _validate_classifier_payload(
            {
                "intent": "entity-lookup",
                "attributes": {
                    "item_query": " show me The Matrix ",
                    "needs_weather_tool": "false",
                    "reason": "lookup_guess",
                },
            }
        )
        self.assertEqual(intent, "entity_lookup")
        self.assertEqual(attributes["item_query"], "show me The Matrix")
        self.assertFalse(attributes["needs_weather_tool"])
        self.assertEqual(attributes["reason"], "lookup_guess")

    def test_validate_classifier_payload_defaults_unknown_intent_and_invalid_attributes(self):
        intent, attributes = _validate_classifier_payload(
            {
                "intent": "totally_new_mode",
                "attributes": "broken",
                "item_query": None,
                "needs_weather_tool": "null",
            }
        )
        self.assertEqual(intent, "general_qa")
        self.assertEqual(attributes["item_query"], "")
        self.assertFalse(attributes["needs_weather_tool"])
        self.assertEqual(attributes["reason"], "ok")

    def test_classifier_routes_project_context_question_to_general_qa(self):
        result = classifier_node(
            {
                "messages": [HumanMessage(content="What recommendation models are available?")],
                "attributes": {},
                "intent": "general_qa",
                "trace_id": "trace-router",
                "result": {},
            },
            config={},
        )
        self.assertEqual(result["intent"], "general_qa")
        self.assertEqual(result["attributes"]["reason"], "project_context_question")

    def test_classifier_routes_backend_vs_recommender_question_to_general_qa(self):
        result = classifier_node(
            {
                "messages": [HumanMessage(content="What is the difference between backend and recommender?")],
                "attributes": {},
                "intent": "general_qa",
                "trace_id": "trace-arch-router",
                "result": {},
            },
            config={},
        )
        self.assertEqual(result["intent"], "general_qa")
        self.assertEqual(result["attributes"]["reason"], "project_context_question")

    def test_classifier_routes_recommendation_follow_up_question_to_general_qa(self):
        result = classifier_node(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "Recommendation follow-up context for this thread:\n"
                            "Previous recommendation request: recommend me sci-fi films\n"
                            "Previous recommendation details: Top recommended items: Arrival; Ex Machina; Blade Runner.\n"
                            "The current user message refers to the previous recommendation list."
                        )
                    ),
                    HumanMessage(content="Which one should I start with first?"),
                ],
                "attributes": {},
                "intent": "general_qa",
                "trace_id": "trace-follow-up-router",
                "result": {},
            },
            config={},
        )
        self.assertEqual(result["intent"], "general_qa")
        self.assertEqual(result["attributes"]["reason"], "recommendation_follow_up_question")

    def test_sanitize_project_context_question_forces_general_qa(self):
        intent, attributes = _sanitize_intent_classification(
            "What is MCP used for in this project?",
            "entity_lookup",
            {"item_query": "MCP used for in this project", "needs_weather_tool": False, "reason": "llm_lookup"},
        )
        self.assertEqual(intent, "general_qa")
        self.assertEqual(attributes["reason"], "sanitized_project_context_question")

    def test_collect_recommendation_supporting_context_keeps_preference_hints(self):
        contexts = _collect_recommendation_supporting_context(
            [
                HumanMessage(content="I like sci-fi and thrillers"),
                HumanMessage(content="recommend me something"),
            ],
            include_refinement_context=True,
        )
        self.assertEqual(len(contexts), 1)
        self.assertIn("I like sci-fi and thrillers", contexts[0])

    def test_deterministic_follow_up_picks_top_recommended_item(self):
        answer = _build_deterministic_recommendation_follow_up_answer(
            [
                SystemMessage(
                    content=(
                        "Recommendation follow-up context for this thread:\n"
                        "Previous recommendation request: recommend me sci-fi films\n"
                        "Previous recommendation details: Top recommended items: Arrival; Ex Machina; Blade Runner.\n"
                        "The current user message refers to the previous recommendation list."
                    )
                ),
                HumanMessage(content="Which one should I start with first?"),
            ],
            "Which one should I start with first?",
        )
        self.assertIsNotNone(answer)
        self.assertIn("Arrival", answer)

    @patch("agent._build_backend_chat_llm")
    @patch("agent.fetch_project_context_sync")
    def test_general_qa_uses_mcp_for_project_context_question(self, mock_fetch_project_context_sync, mock_build_llm):
        mock_fetch_project_context_sync.return_value = "Supported benchmark datasets:\n- MovieLens"

        result = general_qa_node(
            {
                "messages": [HumanMessage(content="What datasets do you support?")],
                "attributes": {},
                "intent": "general_qa",
                "trace_id": "trace-mcp",
                "result": {},
            },
            config={},
        )

        self.assertEqual(result["messages"][0].content, "Supported benchmark datasets:\n- MovieLens")
        mock_fetch_project_context_sync.assert_called_once_with("What datasets do you support?")
        mock_build_llm.assert_not_called()

    @patch("agent._build_backend_chat_llm")
    def test_general_qa_uses_deterministic_follow_up_without_llm(self, mock_build_llm):
        result = general_qa_node(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "Recommendation follow-up context for this thread:\n"
                            "Previous recommendation request: recommend me sci-fi films\n"
                            "Previous recommendation details: Top recommended items: Arrival; Ex Machina; Blade Runner.\n"
                            "The current user message refers to the previous recommendation list."
                        )
                    ),
                    HumanMessage(content="Which one should I start with first?"),
                ],
                "attributes": {},
                "intent": "general_qa",
                "trace_id": "trace-follow-up",
                "result": {},
            },
            config={},
        )
        self.assertIn("Arrival", result["messages"][0].content)
        mock_build_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()

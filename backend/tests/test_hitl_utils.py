import unittest

from hitl_utils import (
    build_follow_up_prompts,
    build_hitl_refinement_context,
    build_recommendation_follow_up_context,
    is_recommendation_context_follow_up,
    is_recommendation_refinement,
)


class HitlUtilsTests(unittest.TestCase):
    def test_refinement_detection_matches_short_follow_up_requests(self):
        self.assertTrue(is_recommendation_refinement("more comedy please"))
        self.assertTrue(is_recommendation_refinement("more like that"))
        self.assertTrue(is_recommendation_refinement("something less mainstream"))
        self.assertFalse(is_recommendation_refinement("recommend me some films"))

    def test_context_follow_up_detection_matches_natural_recommendation_questions(self):
        self.assertTrue(is_recommendation_context_follow_up("Which one should I start with first?"))
        self.assertTrue(is_recommendation_context_follow_up("Why these recommendations?"))
        self.assertFalse(is_recommendation_context_follow_up("What is The Matrix about?"))

    def test_build_hitl_refinement_context_uses_feedback_signal(self):
        context = build_hitl_refinement_context(
            latest_user_prompt="recommend me films",
            latest_assistant_text="I found 10 recommendations in movielens.",
            latest_feedback={"rating": 2, "comment": "Marked as needing work."},
        )
        self.assertIsNotNone(context)
        self.assertIn("Previous recommendation request", context)
        self.assertIn("needing work", context)
        self.assertIn("Treat the current message as a refinement", context)

    def test_build_recommendation_follow_up_context_mentions_previous_list(self):
        context = build_recommendation_follow_up_context(
            latest_user_prompt="recommend me films",
            latest_assistant_text="Top recommended items: Arrival; Ex Machina; Blade Runner.",
            latest_feedback={"rating": 4, "comment": "Good direction."},
        )
        self.assertIsNotNone(context)
        self.assertIn("Previous recommendation details", context)
        self.assertIn("refers to the previous recommendation list", context)

    def test_follow_up_prompts_include_genre_and_control_options(self):
        prompts = build_follow_up_prompts(
            items=[
                {"title": "A", "genres": "Comedy|Romance"},
                {"title": "B", "genres": "Comedy|Drama"},
            ],
            cold_start=False,
        )
        self.assertGreaterEqual(len(prompts), 4)
        self.assertTrue(any("comedy" in prompt.lower() for prompt in prompts))
        self.assertTrue(
            any(
                prompt in prompts
                for prompt in (
                    "Keep the same profile but make the list more diverse",
                    "Give me a more recent version of these recommendations",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()

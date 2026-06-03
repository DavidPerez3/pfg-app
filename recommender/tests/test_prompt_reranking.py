import unittest
from pathlib import Path
import sys

RECOMMENDER_ROOT = Path(__file__).resolve().parents[1]
if str(RECOMMENDER_ROOT) not in sys.path:
    sys.path.insert(0, str(RECOMMENDER_ROOT))

import main as recommender_main


class PromptRerankingTests(unittest.TestCase):
    def test_extract_current_request_prefers_latest_refinement_segment(self):
        prompt = (
            "Previous recommendation request: recommend me some movies\n"
            "Previous assistant recommendation summary: ...\n\n"
            "Current refinement request:\n"
            "more comedy please"
        )
        self.assertEqual(
            recommender_main._extract_current_request(prompt),
            "more comedy please",
        )

    def test_genre_constraints_detect_negative_genre_requests(self):
        include_genres, exclude_genres = recommender_main._genre_constraints_from_prompt(
            "without romance"
        )
        self.assertEqual(include_genres, set())
        self.assertEqual(exclude_genres, {"romance"})

    def test_request_modifiers_treat_less_mainstream_as_niche(self):
        modifiers = recommender_main._request_modifiers("something less mainstream")
        self.assertTrue(modifiers["niche"])
        self.assertFalse(modifiers["mainstream"])


if __name__ == "__main__":
    unittest.main()

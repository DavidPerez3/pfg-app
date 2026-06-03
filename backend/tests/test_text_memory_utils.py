import unittest
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from text_memory_utils import extract_candidate_facts, normalize_fact


class TextMemoryUtilsTests(unittest.TestCase):
    def test_normalize_fact_collapses_whitespace(self):
        self.assertEqual(normalize_fact("  I   like   sci-fi  "), "I like sci-fi")

    def test_extract_candidate_facts_returns_preference_statements(self):
        text = (
            "I like science fiction movies. "
            "I prefer atmospheric thrillers! "
            "My favorite genre is noir."
        )
        facts = extract_candidate_facts(text)
        self.assertIn("I like science fiction movies", facts)
        self.assertIn("I prefer atmospheric thrillers", facts)
        self.assertIn("My favorite genre is noir", facts)

    def test_extract_candidate_facts_deduplicates_matches(self):
        text = "I like jazz. I like jazz."
        facts = extract_candidate_facts(text)
        self.assertEqual(facts, ["I like jazz"])

    def test_extract_candidate_facts_ignores_empty_text(self):
        self.assertEqual(extract_candidate_facts("   "), [])


if __name__ == "__main__":
    unittest.main()

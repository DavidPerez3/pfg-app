import unittest
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent import _classify_intent_fast_path


class AgentFastPathTests(unittest.TestCase):
    def test_preference_statement_is_not_misclassified_as_similarity(self):
        intent, attributes = _classify_intent_fast_path("I like sci-fi and thrillers")
        self.assertEqual(intent, "general_qa")
        self.assertEqual(attributes["reason"], "heuristic_preference_statement")


if __name__ == "__main__":
    unittest.main()

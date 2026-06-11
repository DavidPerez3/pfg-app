import unittest
from pathlib import Path
import sys
import pandas as pd
from unittest.mock import patch

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

    def test_genre_constraints_understand_too_romantic_feedback_style(self):
        include_genres, exclude_genres = recommender_main._genre_constraints_from_prompt(
            "Too romantic and not enough comedy"
        )
        self.assertEqual(include_genres, {"comedy"})
        self.assertEqual(exclude_genres, {"romance"})

    def test_genre_constraints_detect_compound_genre_requests(self):
        include_genres, exclude_genres = recommender_main._genre_constraints_from_prompt(
            "sci-fi and thrillers"
        )
        self.assertEqual(include_genres, {"sci-fi", "thriller"})
        self.assertEqual(exclude_genres, set())

    def test_request_modifiers_treat_less_mainstream_as_niche(self):
        modifiers = recommender_main._request_modifiers("something less mainstream")
        self.assertTrue(modifiers["niche"])
        self.assertFalse(modifiers["mainstream"])

    def test_apply_genre_constraints_to_indices_keeps_exclusions_even_if_list_shrinks(self):
        items_df = pd.DataFrame(
            [
                {"title": "A", "genres": "Romance|Drama"},
                {"title": "B", "genres": "Action|Thriller"},
                {"title": "C", "genres": "Romance|Comedy"},
            ]
        )
        with patch.object(recommender_main, "_load_items", return_value=items_df):
            filtered = recommender_main._apply_genre_constraints_to_indices(
                dataset="movielens",
                candidate_indices=[0, 1, 2],
                include_genres=set(),
                exclude_genres={"romance"},
                top_k=10,
            )
        self.assertEqual(filtered, [1])

    def test_apply_genre_constraints_to_item_ids_keeps_exclusions_but_softens_includes(self):
        with patch.object(
            recommender_main,
            "_dataset_item_text_by_id",
            return_value={
                "1": ("title 1", "romance|drama"),
                "2": ("title 2", "comedy"),
                "3": ("title 3", "thriller"),
            },
        ):
            filtered = recommender_main._apply_genre_constraints_to_item_ids(
                dataset="movielens",
                candidate_item_ids=["1", "2", "3"],
                include_genres={"comedy"},
                exclude_genres={"romance"},
                top_k=3,
            )
        self.assertEqual(filtered, ["2", "3"])

    def test_display_title_from_source_prefers_artist_track_combo_when_needed(self):
        title = recommender_main._display_title_from_source(
            {
                "artist": "Massive Attack",
                "track": "Teardrop",
            },
            "item-1",
        )
        self.assertEqual(title, "Massive Attack - Teardrop")

    def test_display_metadata_from_source_uses_categories_for_non_genre_datasets(self):
        metadata = recommender_main._display_metadata_from_source(
            {
                "categories": "Restaurants, Seafood",
                "city": "Bilbao",
            }
        )
        self.assertEqual(metadata, "Restaurants, Seafood")

    def test_display_title_from_source_falls_back_to_brand_for_sparse_amazon_docs(self):
        title = recommender_main._display_title_from_source(
            {
                "brand": "Sony",
                "category": "Headphones",
            },
            "B000123",
        )
        self.assertEqual(title, "Sony - Headphones")

    def test_display_title_from_row_uses_item_name_when_title_is_blank(self):
        row = pd.Series(
            {
                "title": "",
                "item_name": "B000123",
                "brand": "",
                "category": "",
            }
        )
        title = recommender_main._display_title_from_row(row, "B000123")
        self.assertEqual(title, "B000123")

    def test_search_query_variants_include_normalized_and_split_forms(self):
        variants = recommender_main._search_query_variants("Beyonce - Halo")
        self.assertIn("Beyonce - Halo", variants)
        self.assertIn("Beyonce", variants)
        self.assertIn("Halo", variants)
        self.assertIn("beyonce halo", variants)

    def test_build_elasticsearch_lookup_should_clauses_adds_artist_track_clause(self):
        clauses = recommender_main._build_elasticsearch_lookup_should_clauses(
            "Massive Attack - Teardrop"
        )
        artist_track_clauses = [
            clause for clause in clauses if "bool" in clause and "must" in clause["bool"]
        ]
        self.assertTrue(artist_track_clauses)
        artist_track_clause = artist_track_clauses[0]["bool"]["must"]
        self.assertEqual(artist_track_clause[0]["match_phrase"]["artist"]["query"], "Massive Attack")
        self.assertEqual(artist_track_clause[1]["match_phrase"]["track"]["query"], "Teardrop")

    def test_row_search_text_includes_sparse_metadata_fields(self):
        row = pd.Series(
            {
                "title": "",
                "brand": "Sony",
                "category": "Headphones",
                "city": "Bilbao",
            }
        )
        text = recommender_main._row_search_text(row)
        self.assertIn("Sony", text)
        self.assertIn("Headphones", text)
        self.assertIn("Bilbao", text)

    def test_search_items_from_parquet_matches_artist_and_track_metadata(self):
        items_df = pd.DataFrame(
            [
                {"title": "", "artist_name": "Massive Attack", "track_name": "Teardrop", "genres": ""},
                {"title": "", "artist_name": "Portishead", "track_name": "Roads", "genres": ""},
            ]
        )
        with patch.object(recommender_main, "_load_items", return_value=items_df):
            results = recommender_main._search_items_from_parquet(
                recommender_main.SearchRequest(
                    query="Massive Attack Teardrop",
                    dataset="lastfm",
                    top_k=5,
                )
            )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Massive Attack - Teardrop")

    def test_build_response_uses_row_title_fallback_when_title_column_is_blank(self):
        items_df = pd.DataFrame(
            [
                {"title": "", "item_name": "B000123", "category": "Audio"},
                {"title": "", "item_name": "B000456", "category": "Video"},
            ]
        )
        with patch.object(recommender_main, "_load_items", return_value=items_df):
            response = recommender_main._build_response(
                user_id="u1",
                dataset="amazon_electronics",
                model="matrix_factorization",
                item_indices=[0, 1],
                cold_start=False,
            )
        self.assertEqual(response.items[0].title, "B000123")
        self.assertEqual(response.items[1].title, "B000456")

    def test_run_not_trained_detail_is_consistent_across_model_labels(self):
        self.assertEqual(
            recommender_main._run_not_trained_detail("matrix_factorization", "yelp"),
            "Run not trained: Matrix Factorization is not available for dataset 'yelp'.",
        )
        self.assertEqual(
            recommender_main._run_not_trained_detail("llm_rag", "lastfm"),
            "Run not trained: LLM + RAG is not available for dataset 'lastfm'.",
        )


if __name__ == "__main__":
    unittest.main()

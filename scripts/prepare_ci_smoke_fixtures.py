from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dataset(
    models_root: Path,
    dataset: str,
    items: list[dict[str, object]],
    interactions: list[dict[str, object]],
    user_factors: list[list[float]],
    item_factors: list[list[float]],
) -> None:
    processed_root = models_root / "data" / "processed" / dataset
    processed_root.mkdir(parents=True, exist_ok=True)

    items_path = processed_root / "items.parquet"
    interactions_path = processed_root / "interactions.parquet"

    if not items_path.exists():
        pd.DataFrame(items).to_parquet(items_path, index=False)

    if not interactions_path.exists():
        pd.DataFrame(interactions).to_parquet(interactions_path, index=False)

    weights_root = models_root / "weights" / "matrix_factorization" / dataset
    weights_root.mkdir(parents=True, exist_ok=True)
    user_factors_path = weights_root / "user_factors.npy"
    item_factors_path = weights_root / "item_factors.npy"

    if not user_factors_path.exists():
        np.save(user_factors_path, np.asarray(user_factors, dtype=np.float32))
    if not item_factors_path.exists():
        np.save(item_factors_path, np.asarray(item_factors, dtype=np.float32))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    models_root = repo_root / "pfg-models"
    ensure_dataset(
        models_root=models_root,
        dataset="movielens",
        items=[
            {
                "item_id": "1",
                "title": "The Matrix (1999)",
                "genres_str": "Action|Sci-Fi|Thriller",
                "year": 1999,
            },
            {
                "item_id": "2",
                "title": "Toy Story (1995)",
                "genres_str": "Adventure|Animation|Children|Comedy|Fantasy",
                "year": 1995,
            },
            {
                "item_id": "3",
                "title": "Jumanji (1995)",
                "genres_str": "Adventure|Children|Fantasy",
                "year": 1995,
            },
        ],
        interactions=[
            {"user_id": "u1", "item_id": "1", "rating": 5.0, "timestamp": 915148800},
            {"user_id": "u1", "item_id": "2", "rating": 4.0, "timestamp": 915235200},
            {"user_id": "u2", "item_id": "3", "rating": 5.0, "timestamp": 915321600},
            {"user_id": "u3", "item_id": "1", "rating": 4.0, "timestamp": 915408000},
        ],
        user_factors=[[1.0, 0.2], [0.1, 0.9], [0.8, 0.1]],
        item_factors=[[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]],
    )

    ensure_dataset(
        models_root=models_root,
        dataset="lastfm",
        items=[
            {
                "item_id": "track_1",
                "item_name": "Massive Attack - Teardrop",
                "artist": "Massive Attack",
                "track": "Teardrop",
                "artist_name": "Massive Attack",
                "track_name": "Teardrop",
                "genres": "Trip-Hop",
            },
            {
                "item_id": "track_2",
                "item_name": "Portishead - Roads",
                "artist": "Portishead",
                "track": "Roads",
                "artist_name": "Portishead",
                "track_name": "Roads",
                "genres": "Trip-Hop",
            },
            {
                "item_id": "track_3",
                "item_name": "Radiohead - No Surprises",
                "artist": "Radiohead",
                "track": "No Surprises",
                "artist_name": "Radiohead",
                "track_name": "No Surprises",
                "genres": "Alternative",
            },
        ],
        interactions=[
            {"user_id": "user_1", "item_id": "track_1", "rating": 1.0, "timestamp": 1111111111},
            {"user_id": "user_1", "item_id": "track_2", "rating": 1.0, "timestamp": 1111112222},
            {"user_id": "user_2", "item_id": "track_3", "rating": 1.0, "timestamp": 1111113333},
            {"user_id": "user_3", "item_id": "track_1", "rating": 1.0, "timestamp": 1111114444},
        ],
        user_factors=[[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]],
        item_factors=[[1.0, 0.1], [0.7, 0.3], [0.0, 1.0]],
    )

    print(f"Smoke fixtures ready under: {models_root}")


if __name__ == "__main__":
    main()

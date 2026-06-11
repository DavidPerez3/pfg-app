from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    models_root = repo_root / "pfg-models"
    processed_root = models_root / "data" / "processed" / "movielens"
    processed_root.mkdir(parents=True, exist_ok=True)

    items_path = processed_root / "items.parquet"
    interactions_path = processed_root / "interactions.parquet"

    if not items_path.exists():
        items_df = pd.DataFrame(
            [
                {"item_id": "1", "title": "The Matrix (1999)", "genres_str": "Action|Sci-Fi|Thriller", "year": 1999},
                {"item_id": "2", "title": "Toy Story (1995)", "genres_str": "Adventure|Animation|Children|Comedy|Fantasy", "year": 1995},
                {"item_id": "3", "title": "Jumanji (1995)", "genres_str": "Adventure|Children|Fantasy", "year": 1995},
            ]
        )
        items_df.to_parquet(items_path, index=False)

    if not interactions_path.exists():
        interactions_df = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "1", "rating": 5.0, "timestamp": 915148800},
                {"user_id": "u1", "item_id": "2", "rating": 4.0, "timestamp": 915235200},
                {"user_id": "u2", "item_id": "3", "rating": 5.0, "timestamp": 915321600},
                {"user_id": "u3", "item_id": "1", "rating": 4.0, "timestamp": 915408000},
            ]
        )
        interactions_df.to_parquet(interactions_path, index=False)

    # Optional tiny MF artifacts in case we later extend the smoke path to recommendations.
    weights_root = models_root / "weights" / "matrix_factorization" / "movielens"
    weights_root.mkdir(parents=True, exist_ok=True)
    user_factors_path = weights_root / "user_factors.npy"
    item_factors_path = weights_root / "item_factors.npy"
    if not user_factors_path.exists():
        np.save(user_factors_path, np.asarray([[1.0, 0.2], [0.1, 0.9], [0.8, 0.1]], dtype=np.float32))
    if not item_factors_path.exists():
        np.save(item_factors_path, np.asarray([[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]], dtype=np.float32))

    print(f"Smoke fixtures ready under: {models_root}")


if __name__ == "__main__":
    main()

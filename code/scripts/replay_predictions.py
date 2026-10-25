"""Verify saved predictions against saved models with all target columns removed."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from rec.experiment import score_model
from rec.io import file_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    if not (root / "complete.json").exists():
        raise SystemExit("Replay requires a completed run")
    recipe = json.loads((root / "frozen_recipe.json").read_text("utf-8"))
    config = json.loads((root / "config.json").read_text("utf-8"))
    frame = pd.read_parquet(root / "cache/test.parquet").drop(columns=["label", "query_id"])
    checks = []
    for method in ("manual", "auto", "all_features", "no_relations"):
        for seed in config["model_seeds"]:
            model_path = root / "test" / f"{method}_seed{seed}.cbm"
            predictions_path = root / "test" / f"{method}_seed{seed}_predictions.parquet"
            model = CatBoostRanker().load_model(str(model_path))
            actual = score_model(model, frame, recipe[method]["columns"])
            saved = pd.read_parquet(predictions_path)
            if not np.array_equal(frame[["user_id", "item_id"]], saved[["user_id", "item_id"]]):
                raise ValueError(f"Candidate alignment mismatch: {method}/{seed}")
            error = float(np.max(np.abs(actual - saved.score)))
            if error != 0:
                raise ValueError(f"Prediction replay mismatch: {method}/{seed}, max error {error}")
            checks.append(
                {
                    "method": method,
                    "seed": seed,
                    "rows": len(frame),
                    "max_absolute_error": error,
                    "model_sha256": file_hash(model_path),
                    "predictions_sha256": file_hash(predictions_path),
                }
            )
    write_json(root / "replay_verification.json", {"status": "verified", "checks": checks})
    print(f"Verified {len(checks)} saved models; prediction inputs contained no labels")


if __name__ == "__main__":
    main()

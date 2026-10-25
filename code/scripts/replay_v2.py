"""Replay every saved v2 prediction from models, with targets removed from inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from rec.candidate_ablation import original_pool
from rec.experiment import score_model
from rec.io import file_hash, write_json
from rec.routing import TemporalRouter


def replay(run: Path):
    if not (run / "external/complete.json").exists():
        raise ValueError("External experiment must finish before replay")
    recipe = json.loads((run / "frozen_recipe.json").read_text("utf-8"))
    checks = []
    blocks = (
        ("external", "external_cache/2021-06-21.parquet"),
        ("confirm", "development/2021-08-09.parquet"),
        ("exploratory_test", "development/2021-08-16.parquet"),
    )
    for block, cached in blocks:
        frame = pd.read_parquet(run / cached).drop(columns=["label"])
        output = run / block
        results = json.loads((output / "results.json").read_text("utf-8"))
        paths = sorted(output.rglob("*_predictions.parquet"))
        if len(paths) != len(results["metrics"]):
            raise ValueError(f"Incomplete prediction inventory: {block}")
        manual = {}
        for path in paths:
            method, seed_string = path.stem.removesuffix("_predictions").rsplit("_seed", 1)
            seed = int(seed_string)
            slate = original_pool(frame) if method.endswith("_original_pool") else frame
            base = method.removesuffix("_original_pool")
            if base in ("manual", "hybrid"):
                model_path = path.with_name(path.name.replace("_predictions.parquet", ".cbm"))
                model = CatBoostRanker().load_model(str(model_path))
                actual = score_model(model, slate, recipe[base]["columns"])
                model_hash = file_hash(model_path)
            elif method in ("global_mix", "adaptive_mix", "residual_router"):
                model_path = output / f"{method}_model.json"
                router = TemporalRouter.from_state(json.loads(model_path.read_text("utf-8")))
                if router.offset_column:
                    if seed not in manual:
                        baseline = CatBoostRanker().load_model(
                            str(output / f"manual_seed{seed}.cbm")
                        )
                        manual[seed] = score_model(baseline, frame, recipe["manual"]["columns"])
                    slate = frame.assign(base_score=manual[seed])
                actual = router.predict(slate)
                model_hash = file_hash(model_path)
            else:
                column = {"ease": "ease_l1000_h30", "popularity": "retrieval_pop"}[method]
                actual = frame[column].to_numpy()
                model_hash = None
            saved = pd.read_parquet(path)
            if not np.array_equal(slate[["user_id", "item_id"]], saved[["user_id", "item_id"]]):
                raise ValueError(f"Candidate alignment mismatch: {path}")
            error = float(np.max(np.abs(actual - saved.score.to_numpy())))
            if error > 1e-12:
                raise ValueError(f"Replay mismatch: {path}, max error {error}")
            checks.append(
                {
                    "block": block,
                    "method": method,
                    "seed": seed,
                    "rows": len(slate),
                    "max_absolute_error": error,
                    "model_sha256": model_hash,
                    "predictions_sha256": file_hash(path),
                }
            )
    result = {"status": "verified", "target_columns_removed": ["label"], "checks": checks}
    write_json(run / "replay_verification.json", result)
    print(
        f"Verified {len(checks)} prediction arrays; max error {max(c['max_absolute_error'] for c in checks)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/v2"))
    replay(parser.parse_args().run)

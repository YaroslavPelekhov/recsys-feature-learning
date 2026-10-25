"""Replay v3 models and regenerate full test features after mutating future content."""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

import research_v3 as study
from rec.io import file_hash, write_json
from rec.music import load_music, prepare_music, with_encoder_seed
from rec.transition import DAY


def future_invariance(root):
    events, artists = load_music(root)
    regenerated = []
    destination = root / "future_invariance_replay"
    for day in study.TEST:
        models = destination / "encoders" / f"day{day}"
        models.mkdir(parents=True, exist_ok=True)
        for source in (root / "encoders" / f"day{day}").iterdir():
            target = models / source.name
            if target.exists():
                continue
            if source.suffix == ".npz":
                os.link(source.resolve(), target.resolve())
            elif source.suffix == ".json":
                shutil.copy2(source, target)
        changed = events.copy()
        future = changed.timestamp >= day * DAY - 1800
        changed.loc[future, "item_id"] = np.uint32(2**32 - 1)
        changed.loc[future, "played_ratio_pct"] = 0
        changed.loc[future, "is_organic"] = 0
        changed.loc[future, "artist"] = -999
        rebuilt = prepare_music(changed, artists, day, destination, targets=False)
        original = pd.read_parquet(root / "cache" / f"day{day}.parquet")
        pd.testing.assert_frame_equal(original, rebuilt, check_exact=True)
        regenerated.append(
            {
                "day": day,
                "rows": len(rebuilt),
                "future_rows_mutated": int(future.sum()),
                "all_columns_equal": True,
                "max_absolute_error": 0.0,
                "canonical_cache_sha256": file_hash(root / "cache" / f"day{day}.parquet"),
            }
        )
    return regenerated


def replay(root):
    if not (root / "test/complete.json").exists():
        raise ValueError("Replay requires completed external predictions and metrics")
    recipe = study.load_json(root / "frozen_recipe.json")
    checks = []
    for block, days in (("confirm", (study.CONFIRM,)), ("test", study.TEST)):
        output = root / block
        results = study.load_json(output / "results.json")
        paths = sorted(output.glob("*_predictions.parquet"))
        if len(paths) != len(results["metrics"]):
            raise ValueError("Incomplete prediction inventory")
        for day in days:
            frame = pd.read_parquet(root / "cache" / f"day{day}.parquet").drop(
                columns=["label", "query_id"]
            )
            for path in paths:
                method, suffix = path.stem.removesuffix("_predictions").rsplit("_seed", 1)
                seed, prediction_day = map(int, suffix.split("_day"))
                if day != prediction_day:
                    continue
                if method in ("popularity", "repeat", "userknn"):
                    column = {
                        "popularity": "pop_score",
                        "repeat": "repeat_score",
                        "userknn": "knn_score",
                    }[method]
                    values, model_hash = frame[column].to_numpy(), None
                else:
                    model_path = output / f"{method}_seed{seed}.cbm"
                    model = CatBoostRanker().load_model(str(model_path))
                    values = study.score(
                        model, with_encoder_seed(frame, seed), recipe[method]["columns"]
                    )
                    model_hash = file_hash(model_path)
                saved = pd.read_parquet(path)
                if not np.array_equal(frame[["user_id", "item_id"]], saved[["user_id", "item_id"]]):
                    raise ValueError("Prediction candidate alignment mismatch")
                error = float(np.max(np.abs(values - saved.score.to_numpy())))
                if error > 1e-12:
                    raise ValueError(f"Prediction mismatch: {path}, error {error}")
                checks.append(
                    {
                        "block": block,
                        "method": method,
                        "seed": seed,
                        "day": day,
                        "rows": len(frame),
                        "max_absolute_error": error,
                        "model_sha256": model_hash,
                        "predictions_sha256": file_hash(path),
                    }
                )
    regenerated = future_invariance(root)
    write_json(
        root / "replay_verification.json",
        {
            "status": "verified",
            "removed_columns": ["label", "query_id"],
            "checks": checks,
            "future_invariance": regenerated,
        },
    )
    print(
        f"Verified {len(checks)} predictions and {len(regenerated)} full future-invariance replays"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/v3"))
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()
    if args.features_only:
        checks = future_invariance(args.run)
        result = {
            "status": "verified",
            "future_invariance": checks,
            "main_test_targets_exist": any(
                (args.run / "test" / f"day{day}_truth.parquet").exists() for day in study.TEST
            ),
        }
        write_json(args.run / "future_invariance_pre_metric.json", result)
        print(result)
    else:
        replay(args.run)

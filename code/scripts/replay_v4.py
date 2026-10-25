"""Replay all fixed forecasts and future-content invariants of the final cohort."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

import research_v3 as study
from replay_v3 import future_invariance
from rec.io import file_hash, write_json
from rec.music import load_music
from rec.reconstruction import reconstructed_scores
from rec.transition import DAY, TransitionEmbedding
from support_v3 import fuse


def replay(root):
    if not (root / "test/complete.json").exists():
        raise ValueError("Require completed test")
    recipe = study.load_json(root / "frozen_recipe.json")
    ready = study.load_json(root / "predictions_ready.json")
    events, _ = load_music(root)
    checks, reconstructed = [], []
    for day in study.TEST:
        frame = pd.read_parquet(root / "cache" / f"day{day}.parquet").drop(
            columns=["label", "query_id"]
        )
        changed = events.copy()
        future = changed.timestamp >= day * DAY - 1800
        changed.loc[future, "item_id"] = np.uint32(2**32 - 1)
        changed.loc[future, "played_ratio_pct"] = 0
        changed.loc[future, "artist"] = -999
        history = changed[changed.timestamp < day * DAY - 1800]
        for seed in study.SEEDS:
            encoder = TransitionEmbedding.load(
                root / "encoders" / f"day{day}" / f"global_seed{seed}.npz"
            )
            scores, _ = reconstructed_scores(frame, history, encoder, day * DAY - 1800)
            cached = pd.read_parquet(root / "scores" / f"day{day}_seed{seed}.parquet")
            pd.testing.assert_frame_equal(scores, cached, check_exact=True)
            reconstructed.append(
                {
                    "day": day,
                    "seed": seed,
                    "max_absolute_error": 0.0,
                    "future_rows_mutated": int(future.sum()),
                }
            )
            model = CatBoostRanker().load_model(str(root / "test" / f"manual_seed{seed}.cbm"))
            manual = study.score(model, frame, recipe["manual"]["columns"])
            for method in (
                "manual",
                "legacy",
                "G0",
                "G1",
                "A0",
                "A1",
                "fusionG0",
                "fusionG1",
                "fusionA0",
                "fusionA1",
            ):
                if method == "manual":
                    values = manual
                elif method.startswith("fusion"):
                    variant = method.removeprefix("fusion")
                    values = fuse(
                        scores,
                        manual,
                        scores[variant].to_numpy(),
                        recipe["fusion"][variant]["alpha"],
                    )
                else:
                    values = scores[method].to_numpy()
                path = root / "test" / f"{method}_seed{seed}_day{day}_predictions.parquet"
                saved = pd.read_parquet(path)
                error = float(np.max(np.abs(values - saved.score.to_numpy())))
                if error > 1e-12 or file_hash(path) != ready["predictions"][path.name]:
                    raise ValueError(f"Forecast replay mismatch: {path}")
                checks.append(
                    {
                        "method": method,
                        "seed": seed,
                        "day": day,
                        "max_absolute_error": error,
                        "predictions_sha256": file_hash(path),
                    }
                )
        for method, column in (
            ("repeat", "repeat_score"),
            ("popularity", "pop_score"),
            ("userknn", "knn_score"),
        ):
            path = root / "test" / f"{method}_seed0_day{day}_predictions.parquet"
            saved = pd.read_parquet(path)
            error = float(np.max(np.abs(frame[column].to_numpy() - saved.score.to_numpy())))
            if error > 1e-12 or file_hash(path) != ready["predictions"][path.name]:
                raise ValueError("Control replay mismatch")
            checks.append(
                {
                    "method": method,
                    "seed": 0,
                    "day": day,
                    "max_absolute_error": error,
                    "predictions_sha256": file_hash(path),
                }
            )
    full = future_invariance(root)
    write_json(
        root / "replay_verification.json",
        {
            "status": "verified",
            "removed_columns": ["label", "query_id"],
            "checks": checks,
            "reconstruction_future_invariance": reconstructed,
            "full_candidate_future_invariance": full,
        },
    )
    print(
        "VERIFIED",
        len(checks),
        "forecast arrays,",
        len(reconstructed),
        "reconstruction and",
        len(full),
        "full candidate replays",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/v4"))
    replay(parser.parse_args().run)

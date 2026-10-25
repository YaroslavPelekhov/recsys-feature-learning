"""Post-test descriptive audit of discovery coverage and saved top-10 exposure."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import research_v3 as study
from rec.io import write_json
from rec.metrics import user_metrics
from rec.transition import DAY


def pair_index(frame):
    return pd.MultiIndex.from_frame(frame[["user_id", "item_id"]])


def audit(root):
    if not (root / "test/complete.json").exists():
        raise ValueError("Exposure audit is descriptive and requires completed test")
    events = pd.read_parquet(
        root / "sampled_events.parquet", columns=["uid", "item_id", "timestamp"]
    ).rename(columns={"uid": "user_id"})
    coverage, exposure = [], []
    for day in study.TEST:
        history = events[events.timestamp < day * DAY - 1800]
        known = pair_index(history.drop_duplicates(["user_id", "item_id"]))
        truth = pd.read_parquet(root / "test" / f"day{day}_truth.parquet")
        candidates = pd.read_parquet(
            root / "cache" / f"day{day}.parquet", columns=["user_id", "item_id"]
        )
        candidates = candidates[candidates.user_id.isin(truth.user_id)].reset_index(drop=True)
        novel_candidates = ~pair_index(candidates).isin(known)
        novel_truth = truth[~pair_index(truth).isin(known)]
        reachable = pair_index(novel_truth).isin(pair_index(candidates))
        per_user = user_metrics(
            candidates,
            np.zeros(len(candidates)),
            novel_truth,
            np.sort(novel_truth.user_id.unique()),
        )
        coverage.append(
            {
                "day": day,
                "evaluated_users": truth.user_id.nunique(),
                "novel_target_users": novel_truth.user_id.nunique(),
                "novel_targets": len(novel_truth),
                "reachable_novel_targets": int(reachable.sum()),
                "candidate_novel_pair_share": float(novel_candidates.mean()),
                "novel_candidate_recall_user_macro": float(per_user.candidate_recall.mean()),
            }
        )
        for path in sorted((root / "test").glob(f"*_day{day}_predictions.parquet")):
            method, suffix = path.stem.removesuffix("_predictions").rsplit("_seed", 1)
            seed = int(suffix.split("_day")[0])
            predictions = pd.read_parquet(path)
            predictions = predictions[predictions.user_id.isin(truth.user_id)]
            top = (
                predictions.sort_values(
                    ["user_id", "score", "item_id"], ascending=[True, False, True], kind="stable"
                )
                .groupby("user_id", sort=False)
                .head(10)
            )
            novel = ~pair_index(top).isin(known)
            exposure.append(
                {
                    "day": day,
                    "method": method,
                    "seed": seed,
                    "top10_pairs": len(top),
                    "novel_top10_pairs": int(novel.sum()),
                    "novel_top10_share": float(novel.mean()),
                    "novel_top10_hits": int(
                        pair_index(top[novel]).isin(pair_index(novel_truth)).sum()
                    ),
                }
            )
    output = root / "exposure_audit"
    output.mkdir(exist_ok=True)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(exposure).to_csv(output / "exposure.csv", index=False)
    write_json(
        output / "audit.json",
        {
            "scope": "descriptive post-test; no model selection",
            "coverage": coverage,
            "exposure": exposure,
        },
    )
    print(pd.DataFrame(coverage).to_string(index=False))
    print(
        pd.DataFrame(exposure)
        .groupby("method")[["novel_top10_share", "novel_top10_hits"]]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/v3"))
    audit(parser.parse_args().run)

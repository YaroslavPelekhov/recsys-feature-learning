"""Compare fitted-pair discrimination with later old-search transitions; no v4 labels."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audit_critic_v3 import pair_logits
from rec.io import file_hash, write_json
from rec.transition import DAY, TransitionEmbedding, permuted_targets, transition_pairs


def audit(base, output):
    events = pd.read_parquet(base / "sampled_events.parquet").rename(columns={"uid": "user_id"})
    train = transition_pairs(events[events.timestamp < 292 * DAY - 1800], 292 * DAY - 1800)
    future = transition_pairs(events[events.timestamp < 295 * DAY], 295 * DAY)
    future = future[future.day == 294].reset_index(drop=True)
    source_role, target_role = set(train.source), set(train.target)
    supported = future.source.isin(source_role) & future.target.isin(target_role)
    held = future[supported].reset_index(drop=True)
    known = pd.MultiIndex.from_frame(train[["source", "target"]].drop_duplicates())
    seen = pd.MultiIndex.from_frame(held[["source", "target"]]).isin(known)
    rows = []
    for name in ("global", "conditional"):
        model = TransitionEmbedding.load(base / "encoders/day292" / f"{name}_seed17.npz")
        for block, pairs in (("training", train), ("later_day294", held)):
            real = pair_logits(model, pairs.source, pairs.target)
            for conditional in (False, True):
                negative_targets = permuted_targets(pairs, 719003, conditional)
                negative = pair_logits(model, pairs.source, negative_targets)
                auc = roc_auc_score(
                    np.r_[np.ones(len(pairs)), np.zeros(len(pairs))], np.r_[real, negative]
                )
                loss = 0.5 * (
                    np.logaddexp(0, -real.astype(float)).mean()
                    + np.logaddexp(0, negative.astype(float)).mean()
                )
                rows.append(
                    {
                        "encoder": name,
                        "period": block,
                        "noise": "conditional" if conditional else "global",
                        "positive_pairs": len(pairs),
                        "auc": float(auc),
                        "balanced_loss": float(loss),
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "temporal_critic.csv", index=False)
    write_json(
        output / "temporal_critic.json",
        {
            "scope": "mechanistic diagnostic on old v3 search, not a new external test",
            "fit_history_exclusive": 292 * DAY - 1800,
            "evaluation_target_day": 294,
            "future_pairs": len(future),
            "supported_pairs": len(held),
            "supported_fraction": float(supported.mean()),
            "past_seen_edge_fraction_on_supported_pairs": float(seen.mean()),
            "v4_targets_used": False,
            "rows": rows,
            "models": {
                name: file_hash(base / "encoders/day292" / f"{name}_seed17.npz")
                for name in ("global", "conditional")
            },
        },
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print("Role-supported fraction", supported.mean(), "past-seen edge fraction", seen.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/v3"))
    parser.add_argument("--output", type=Path, default=Path("runs/v4/audits"))
    args = parser.parse_args()
    audit(args.base, args.output)

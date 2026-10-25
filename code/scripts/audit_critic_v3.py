"""Historical training-pair diagnostics; never uses recommendation target sets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rec.io import write_json
from rec.transition import DAY, TransitionEmbedding, noise_targets, transition_pairs


def pair_logits(model, source, target, batch_size=65536):
    source_rows = model.items.get_indexer(source)
    target_rows = model.items.get_indexer(target)
    if (source_rows < 0).any() or (target_rows < 0).any():
        raise ValueError("Training pair absent from encoder vocabulary")
    result = np.empty(len(source), dtype=np.float32)
    for start in range(0, len(source), batch_size):
        end = start + batch_size
        result[start:end] = np.sum(
            model.source[source_rows[start:end]] * model.target[target_rows[start:end]], axis=1
        ) / np.sqrt(model.source.shape[1])
    return result


def audit(base, support, output):
    events = pd.read_parquet(base / "sampled_events.parquet").rename(columns={"uid": "user_id"})
    cutoff = 294 * DAY - 1800
    pairs = transition_pairs(events[events.timestamp < cutoff], cutoff)
    noise = {
        name: noise_targets(pairs, 190019, kind)
        for name, kind in (("global", False), ("conditional", True), ("mixture", "mixture"))
    }
    rows = []
    for name, root in (("global", base), ("conditional", base), ("mixture", support)):
        model = TransitionEmbedding.load(root / "encoders/day294" / f"{name}_seed17.npz")
        positive = pair_logits(model, pairs.source, pairs.target)
        for distribution, targets in noise.items():
            negative = pair_logits(model, pairs.source, targets)
            loss = 0.5 * (
                np.logaddexp(0, -positive.astype(np.float64)).mean()
                + np.logaddexp(0, negative.astype(np.float64)).mean()
            )
            auc = roc_auc_score(
                np.r_[np.ones(len(pairs)), np.zeros(len(pairs))], np.r_[positive, negative]
            )
            rows.append(
                {
                    "encoder": name,
                    "noise_distribution": distribution,
                    "balanced_logistic_loss_without_regularizer": float(loss),
                    "training_pair_auc": float(auc),
                    "real_logit_mean": float(positive.mean()),
                    "noise_logit_mean": float(negative.mean()),
                    "noise_positive_logit_share": float((negative > 0).mean()),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "critic_audit.csv", index=False)
    write_json(
        output / "critic_audit.json",
        {
            "scope": "search prefix; positive pairs reused from encoder fitting, not held out",
            "noise_permutation_seed": 190019,
            "cutoff_exclusive": cutoff,
            "pairs": len(pairs),
            "held_out_target_sets_used": False,
            "rows": rows,
        },
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/v3"))
    parser.add_argument("--support", type=Path, default=Path("runs/v3_support"))
    parser.add_argument("--output", type=Path, default=Path("runs/v3_support"))
    args = parser.parse_args()
    audit(args.base, args.support, args.output)

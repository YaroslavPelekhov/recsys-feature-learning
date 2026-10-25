"""Inspect high-scoring development pairs; no held-out target labels are accessed."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_noise_support import expected_null
from rec.io import write_json
from rec.music import load_music, bounded_top
from rec.transition import DAY, TransitionEmbedding, transition_pairs


def audit(base, support, output):
    events, artists = load_music(base)
    pairs = transition_pairs(events[events.timestamp < 294 * DAY - 1800], 294 * DAY - 1800)
    items, counts, background = expected_null(pairs)
    frequencies = pairs.source.value_counts().sort_values(ascending=False, kind="stable")
    source_items = frequencies.head(50).index.to_numpy()
    rows = []
    for name, root in (("global", base), ("conditional", base), ("mixture", support)):
        model = TransitionEmbedding.load(root / "encoders/day294" / f"{name}_seed17.npz")
        locations = model.items.get_indexer(source_items)
        values = model.retrieval_scores(model.source[locations])
        for source, vector in zip(source_items, values):
            vector[model.items.get_loc(source)] = -np.inf
            chosen = bounded_top(vector, model.items.to_numpy(), 10)
            destinations = model.items.to_numpy()[chosen]
            x = items.get_indexer(np.full(len(chosen), source))
            y = items.get_indexer(destinations)
            observed = np.asarray(counts[x, y]).ravel()
            expected = np.asarray(background[x, y]).ravel()
            for rank, (target, score, observed_count, expected_count) in enumerate(
                zip(destinations, vector[chosen], observed, expected)
            ):
                rows.append(
                    {
                        "method": name,
                        "source": int(source),
                        "target": int(target),
                        "rank": rank + 1,
                        "score": float(score),
                        "same_primary_artist": bool(
                            artists.get(source, -1) == artists.get(target, -2)
                        ),
                        "real_transition_count": float(observed_count),
                        "context_expected_count": float(expected_count),
                        "outside_context_support": bool(expected_count == 0),
                    }
                )
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "neighbor_audit.csv", index=False)
    summary = (
        table.groupby("method")
        .agg(
            pairs=("target", "size"),
            same_artist_share=("same_primary_artist", "mean"),
            outside_context_support_share=("outside_context_support", "mean"),
            mean_observed_count=("real_transition_count", "mean"),
            mean_expected_count=("context_expected_count", "mean"),
        )
        .reset_index()
    )
    write_json(
        output / "neighbor_audit.json",
        {
            "scope": "search snapshot only, top50 historical source items, self excluded, top10 targets",
            "summary": summary.to_dict("records"),
        },
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/v3"))
    parser.add_argument("--support", type=Path, default=Path("runs/v3_support"))
    parser.add_argument("--output", type=Path, default=Path("runs/v3_support"))
    args = parser.parse_args()
    audit(args.base, args.support, args.output)

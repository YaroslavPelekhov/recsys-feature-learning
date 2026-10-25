"""Development-only search-fold probe; never opens test labels."""

import json
import logging
import time
from pathlib import Path

import pandas as pd

from rec.data import load_sample
from rec.io import write_json
from rec.linear import HistoricalMatrix
from rec.metrics import summarize, user_metrics
from rec.protocol import history_before


def main():
    logging.basicConfig(level=logging.INFO)
    run = Path("runs/main")
    events, _, _, _ = load_sample(Path("data/raw"), 10000, 2026)
    audit = json.loads((run / "cache/search_audit.json").read_text("utf-8"))
    cutoff = pd.Timestamp(audit["cutoff"])
    matrix = HistoricalMatrix.build(history_before(events, cutoff), cutoff)
    pairs = pd.read_parquet(run / "cache/search.parquet")
    truth = pd.read_parquet(run / "cache/search_truth.parquet")
    rows = []
    for regularization in (1, 10, 100, 1000):
        start = time.perf_counter()
        coefficients = matrix.ease(regularization)
        for half_life in (None, 7, 30):
            scores = matrix.pair_scores(pairs, coefficients, half_life)
            metrics = user_metrics(pairs, scores, truth, truth.user_id.unique())
            rows.append(
                {
                    "method": "EASE",
                    "lambda": regularization,
                    "half_life": half_life,
                    **summarize(metrics),
                    "seconds": time.perf_counter() - start,
                }
            )
            print(rows[-1], flush=True)
    coefficients = matrix.rp3beta()
    scores = matrix.pair_scores(pairs, coefficients)
    rows.append(
        {
            "method": "RP3beta",
            **summarize(user_metrics(pairs, scores, truth, truth.user_id.unique())),
        }
    )
    print(rows[-1], flush=True)
    write_json(Path("runs/development/linear_probe.json"), rows)


if __name__ == "__main__":
    main()

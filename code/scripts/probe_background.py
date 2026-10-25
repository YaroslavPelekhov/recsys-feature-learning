"""Search-only feasibility probe with all historical background users."""

import logging
import time
import zipfile
from pathlib import Path

import pandas as pd

from rec.linear import HistoricalMatrix
from rec.metrics import summarize, user_metrics
from rec.io import write_json


def main():
    logging.basicConfig(level=logging.INFO)
    with zipfile.ZipFile("data/raw/data_original.zip") as archive:
        with archive.open("data_original/interactions.csv") as source:
            events = pd.read_csv(
                source,
                usecols=["user_id", "item_id", "last_watch_dt"],
                dtype={"user_id": "int32", "item_id": "int32"},
            )
    events = events.rename(columns={"last_watch_dt": "timestamp"})
    events["timestamp"] = pd.to_datetime(events.timestamp)
    cutoff = pd.Timestamp("2021-08-02")
    historical = events[events.timestamp < cutoff]
    start = time.perf_counter()
    matrix = HistoricalMatrix.build(historical, cutoff)
    print("matrix", matrix.binary.shape, matrix.binary.nnz, flush=True)
    coefficients = matrix.ease(1000, threads=12)
    fit_seconds = time.perf_counter() - start
    print("fit_seconds", fit_seconds, flush=True)
    pairs = pd.read_parquet(Path("runs/main/cache/search.parquet"))
    truth = pd.read_parquet(Path("runs/main/cache/search_truth.parquet"))
    rows = []
    for half_life in (None, 7, 30):
        scores = matrix.pair_scores(pairs, coefficients, half_life)
        metrics = summarize(user_metrics(pairs, scores, truth, truth.user_id.unique()))
        rows.append({"half_life": half_life, **metrics})
        print(half_life, metrics, flush=True)
    write_json(
        Path("runs/development/background_probe.json"),
        {
            "users": len(matrix.users),
            "items": len(matrix.items),
            "edges": matrix.binary.nnz,
            "fit_seconds": fit_seconds,
            "lambda": 1000,
            "metrics": rows,
            "scope": "search-only feasibility; more background users than main comparison",
        },
    )


if __name__ == "__main__":
    main()

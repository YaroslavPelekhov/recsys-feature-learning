"""Search-only comparison of learned graph features against the original manual baseline."""

import json
import logging
from pathlib import Path

import pandas as pd

from rec.data import load_sample
from rec.experiment import tune_trial
from rec.io import write_json
from rec.linear import HistoricalMatrix
from rec.protocol import Config, history_before


def main():
    logging.basicConfig(level=logging.INFO)
    root = Path("runs/main")
    output = Path("runs/development/hybrid")
    output.mkdir(parents=True, exist_ok=True)
    events, _, _, _ = load_sample(Path("data/raw"), 10000, 2026)
    prepared = {}
    learned = []
    for fold in ("train_0", "train_1", "train_2", "train_3", "search"):
        path = output / f"{fold}.parquet"
        if path.exists():
            prepared[fold] = pd.read_parquet(path)
            learned = [name for name in prepared[fold] if name.startswith("ease_")]
            continue
        frame = pd.read_parquet(root / "cache" / f"{fold}.parquet")
        audit = json.loads((root / "cache" / f"{fold}_audit.json").read_text("utf-8"))
        cutoff = pd.Timestamp(audit["cutoff"])
        matrix = HistoricalMatrix.build(history_before(events, cutoff), cutoff)
        for regularization in (100, 1000):
            coefficients = matrix.ease(regularization)
            for half_life in (None, 7, 30):
                name = f"ease_l{regularization}_h{half_life}"
                frame[name] = matrix.pair_scores(frame, coefficients, half_life)
        learned = [name for name in frame if name.startswith("ease_")]
        frame.to_parquet(path, index=False)
        prepared[fold] = frame
        print("prepared", fold, flush=True)
    columns = json.loads((root / "frozen_recipe.json").read_text("utf-8"))["manual"]["columns"]
    train = pd.concat(
        [frame for name, frame in prepared.items() if name.startswith("train")], ignore_index=True
    )
    truth = pd.read_parquet(root / "cache/search_truth.parquet")
    config = Config(iterations=350, model_seeds=(17,))
    result = tune_trial(train, prepared["search"], truth, columns + learned, config, 7)
    print("HYBRID", result, flush=True)
    write_json(output / "search.json", result)


if __name__ == "__main__":
    main()

"""Secondary, pre-test support-mixture correction; the original v3 recipe stays frozen."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import research_v3 as study
from parallel_v3 import model_frame
from rec.experiment import measurement
from rec.io import file_hash, write_json
from rec.metrics import paired_bootstrap, summarize, user_metrics
from rec.music import load_music
from rec.transition import (
    DAY,
    HALF_LIVES,
    TransitionEmbedding,
    historical_profiles,
    transition_pairs,
)


def augment(base, root, days):
    events, _ = load_music(base)
    logs = []
    for day in days:
        destination = root / "cache" / f"day{day}.parquet"
        if destination.exists():
            continue
        if day in study.TEST and any(
            (base / "test" / f"day{d}_truth.parquet").exists() for d in study.TEST
        ):
            raise ValueError("Support features must be fixed before main held-out targets")
        cutoff = day * DAY - 1800
        history = events[events.timestamp < cutoff]
        users = pd.Index(np.sort(history.user_id.unique()))
        items = pd.Index(np.sort(history.item_id.unique()))
        profiles = historical_profiles(history, users, items, cutoff)
        pairs = transition_pairs(history, cutoff)
        frame = pd.read_parquet(
            base / "cache" / f"day{day}.parquet", columns=["user_id", "item_id"]
        )
        row_users = users.get_indexer(frame.user_id)
        for seed in study.SEEDS:
            path = root / "encoders" / f"day{day}" / f"mixture_seed{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            if path.exists() and path.with_suffix(".json").exists():
                encoder = TransitionEmbedding.load(path)
                info = study.load_json(path.with_suffix(".json"))
            else:
                encoder = TransitionEmbedding.fit(pairs, "mixture", seed)
                encoder.save(path)
                info = {
                    "metadata": encoder.metadata,
                    "losses": encoder.losses,
                    "seconds": time.perf_counter() - start,
                }
                write_json(path.with_suffix(".json"), info)
            logs.append({"day": day, "seed": seed, **info})
            for half in HALF_LIVES:
                vectors = encoder.user_vectors(profiles[half], items)
                frame[f"mixture_seed{seed}_h{half}"] = encoder.pair_scores(
                    vectors, row_users, frame.item_id.to_numpy()
                )
            print(f"SUPPORT day{day}/{seed}: {info['seconds']:.1f}s", flush=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
        write_json(
            destination.with_suffix(".json"),
            {
                "base_cache_sha256": file_hash(base / "cache" / f"day{day}.parquet"),
                "history_max_timestamp": int(history.timestamp.max()),
                "history_cutoff_exclusive": cutoff,
                "target_columns": False,
            },
        )
    if logs:
        previous = (
            study.load_json(root / "encoder_training.json")
            if (root / "encoder_training.json").exists()
            else []
        )
        write_json(root / "encoder_training.json", previous + logs)


def standardize(scores, users):
    ids, _ = pd.factorize(users, sort=False)
    counts = np.bincount(ids)
    means = np.bincount(ids, weights=scores) / counts
    centered = np.asarray(scores) - means[ids]
    variance = np.bincount(ids, weights=centered**2) / counts
    return centered / np.maximum(np.sqrt(variance[ids]), 1e-6)


def fuse(frame, manual, learned, alpha):
    if alpha == 0:
        return np.asarray(manual).copy()
    return standardize(manual, frame.user_id) + alpha * standardize(learned, frame.user_id)


def development(base, root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_recipe.json").exists():
        raise FileExistsError("Support recipe already frozen")
    if any((base / "test" / f"day{day}_truth.parquet").exists() for day in study.TEST):
        raise ValueError("Support extension must be fixed before held-out targets")
    write_json(
        root / "protocol.json",
        {
            "amendment_sha256": file_hash(Path("docs/V3_SUPPORT_AMENDMENT.md")),
            "base_protocol_sha256": file_hash(base / "protocol.json"),
            "sample_sha256": file_hash(base / "sample.json"),
            "main_test_targets_exist": False,
            "primary_status": "secondary extension after search diagnostics",
        },
    )
    resources = []
    with measurement(resources, "prepare/development"):
        augment(base, root, (*study.TRAIN, study.SEARCH, study.CONFIRM))
    manual_columns = study.load_json(base / "cache/groups.json")["manual"]
    options = [
        row for row in study.load_json(base / "search_trials.json") if row["method"] == "manual"
    ]
    if len(options) != 2:
        raise ValueError("Both manual HPO depths must finish before support tuning")
    spec = max(options, key=lambda row: row["ndcg@10"])
    with measurement(resources, "manual_search_refit"):
        training = pd.concat(
            [model_frame(base, day, manual_columns, 17) for day in study.TRAIN], ignore_index=True
        )
        model = study.fit_model(training, manual_columns, spec["depth"], spec["iterations"], 17)
        model.save_model(str(root / "manual_search.cbm"))
        search = model_frame(base, study.SEARCH, manual_columns, 17)
        manual = study.score(model, search, manual_columns)
    learned = pd.read_parquet(root / "cache" / f"day{study.SEARCH}.parquet")
    if not np.array_equal(search[["user_id", "item_id"]], learned[["user_id", "item_id"]]):
        raise ValueError("Support search candidate alignment mismatch")
    truth = pd.read_parquet(base / "cache" / f"day{study.SEARCH}_truth.parquet")
    trials = []
    for half in HALF_LIVES:
        for alpha in (0.0, 0.05, 0.1, 0.25, 0.5):
            values = fuse(search, manual, learned[f"mixture_seed17_h{half}"].to_numpy(), alpha)
            score = float(study.metric(search, values, truth).ndcg.mean())
            trials.append({"half_life_hours": half, "alpha": alpha, "ndcg@10": score})
    write_json(root / "search_trials.json", trials)
    best = max(trials, key=lambda row: row["ndcg@10"])
    recipe = {
        **best,
        "manual": {
            "depth": spec["depth"],
            "iterations": spec["iterations"],
            "columns": manual_columns,
        },
        "mixture_global_weight": 0.5,
        "seeds": study.SEEDS,
        "protocol_sha256": file_hash(root / "protocol.json"),
        "base_protocol_sha256": file_hash(base / "protocol.json"),
        "external_labels_used": False,
    }
    write_json(root / "frozen_recipe.json", recipe)
    write_json(root / "development_resources.json", resources)
    print("SUPPORT FROZEN", best, flush=True)


def wait_for(path, started):
    while not path.exists():
        if time.monotonic() - started > 7200:
            raise TimeoutError(f"Dependency did not complete: {path}")
        time.sleep(5)


def predict(base, root):
    recipe = study.load_json(root / "frozen_recipe.json")
    started = time.monotonic()
    wait_for(base / "frozen_recipe.json", started)
    manual_spec = study.load_json(base / "frozen_recipe.json")["manual"]
    if any(manual_spec[key] != recipe["manual"][key] for key in ("depth", "iterations", "columns")):
        raise ValueError("Manual prior differs from the frozen support specification")
    for day in study.TEST:
        wait_for(base / "cache" / f"day{day}.parquet", started)
    resources = []
    with measurement(resources, "prepare/test"):
        augment(base, root, study.TEST)
    output = root / "test"
    output.mkdir(exist_ok=True)
    for day in (*study.TEST, study.CONFIRM):
        features = pd.read_parquet(root / "cache" / f"day{day}.parquet")
        seeds = study.SEEDS if day in study.TEST else (17,)
        for seed in seeds:
            block = "test" if day in study.TEST else "confirm"
            path = base / block / f"manual_seed{seed}_day{day}_predictions.parquet"
            wait_for(path, started)
            manual = pd.read_parquet(path)
            if not np.array_equal(manual[["user_id", "item_id"]], features[["user_id", "item_id"]]):
                raise ValueError("Support inference alignment mismatch")
            if any((base / "test" / f"day{d}_truth.parquet").exists() for d in study.TEST):
                raise ValueError("Support predictions were not fixed before main test labels")
            values = fuse(
                features,
                manual.score.to_numpy(),
                features[f"mixture_seed{seed}_h{recipe['half_life_hours']}"].to_numpy(),
                recipe["alpha"],
            )
            manual[["user_id", "item_id"]].assign(score=values).to_parquet(
                output / f"support_seed{seed}_day{day}_predictions.parquet", index=False
            )
            print(f"SUPPORT predicted day{day}/seed{seed} before labels", flush=True)
    write_json(root / "test_resources.json", resources)
    write_json(
        root / "predictions_ready.json",
        {
            "recipe_sha256": file_hash(root / "frozen_recipe.json"),
            "base_recipe_sha256": file_hash(base / "frozen_recipe.json"),
            "main_test_targets_exist": False,
            "predictions": {
                p.name: file_hash(p) for p in sorted(output.glob("*_predictions.parquet"))
            },
        },
    )


def evaluate(base, root):
    ready = study.load_json(root / "predictions_ready.json")
    if ready["main_test_targets_exist"] or ready["recipe_sha256"] != file_hash(
        root / "frozen_recipe.json"
    ):
        raise ValueError("Invalid pre-test support freeze")
    if not (base / "test/complete.json").exists():
        raise ValueError("Main test must complete before support evaluation")
    tables, rows = [], []
    for path in sorted((root / "test").glob("*_predictions.parquet")):
        if file_hash(path) != ready["predictions"][path.name]:
            raise ValueError("Support predictions changed after freeze")
        suffix = path.stem.removesuffix("_predictions").split("_seed")[1]
        seed, day = map(int, suffix.split("_day"))
        truth = pd.read_parquet(
            base / ("test" if day in study.TEST else "cache") / f"day{day}_truth.parquet"
        )
        predictions = pd.read_parquet(path)
        predictions = predictions[predictions.user_id.isin(truth.user_id)].reset_index(drop=True)
        metrics = user_metrics(
            predictions, predictions.score.to_numpy(), truth, np.sort(truth.user_id.unique())
        )
        rows.append({"day": day, "seed": seed, **summarize(metrics)})
        metrics["day"] = day
        metrics.to_parquet(
            root / "test" / f"support_seed{seed}_day{day}_users.parquet", index=False
        )
        if day in study.TEST:
            tables.append(metrics)
    user_day = pd.concat(tables).groupby(["user_id", "day"], as_index=False).mean(numeric_only=True)
    means = user_day.drop(columns="day").groupby("user_id", as_index=False).mean(numeric_only=True)
    means.to_parquet(root / "test/support_users.parquet", index=False)
    manual = pd.read_parquet(base / "test/manual_users.parquet")
    result = {
        "metrics": rows,
        "overall": summarize(means),
        "paired_vs_manual": paired_bootstrap(manual, means, 2028),
        "status": "secondary pre-test extension, not main primary",
    }
    write_json(root / "test/results.json", result)
    write_json(
        root / "test/complete.json",
        {
            "results_sha256": file_hash(root / "test/results.json"),
            "recipe_sha256": file_hash(root / "frozen_recipe.json"),
        },
    )
    print("SUPPORT RESULT", result["overall"], result["paired_vs_manual"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["development", "predict", "evaluate"])
    parser.add_argument("--base", type=Path, default=Path("runs/v3"))
    parser.add_argument("--root", type=Path, default=Path("runs/v3_support"))
    args = parser.parse_args()
    {"development": development, "predict": predict, "evaluate": evaluate}[args.phase](
        args.base, args.root
    )

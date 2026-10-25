"""Yambda study: conditional-null transition embeddings versus matched global noise."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from rec.experiment import measurement
from rec.io import environment, file_hash, write_json
from rec.metrics import paired_bootstrap, summarize, user_metrics
from rec.music import MUSIC_CATS, load_music, music_truth, prepare_music, with_encoder_seed
from rec.transition import HALF_LIVES

TRAIN = (286, 288, 290, 292)
SEARCH, CONFIRM, TEST = 294, 296, (298, 299)
SEEDS = (17, 42, 73)


def load_json(path):
    return json.loads(path.read_text("utf-8"))


def columns_for(method, manual):
    raw = [f"raw_h{h}" for h in HALF_LIVES]
    signed = [f"signed_h{h}" for h in HALF_LIVES]
    global_columns = [f"global_h{h}" for h in HALF_LIVES]
    conditional = [f"conditional_h{h}" for h in HALF_LIVES]
    extra = {
        "manual": [],
        "raw": raw,
        "signed": signed,
        "global": global_columns,
        "conditional": conditional,
        "all": raw + signed + global_columns + conditional,
    }[method]
    return manual + extra


def fit_model(frame, columns, depth, iterations, seed):
    mixed = frame.groupby("query_id").label.transform("nunique") > 1
    train = frame.loc[mixed]
    pool = Pool(train[columns], train.label, group_id=train.query_id, cat_features=MUSIC_CATS)
    model = CatBoostRanker(
        depth=depth,
        iterations=iterations,
        learning_rate=0.05,
        loss_function="YetiRank:mode=NDCG;top=10",
        l2_leaf_reg=5,
        random_seed=seed,
        thread_count=12,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(pool)
    return model


def prediction_pool(frame, columns):
    return Pool(frame[columns], cat_features=MUSIC_CATS)


def score(model, frame, columns):
    return model.predict(prediction_pool(frame, columns))


def metric(frame, scores, truth):
    return user_metrics(frame, np.asarray(scores), truth, np.sort(truth.user_id.unique()))


def development(root):
    if (root / "frozen_recipe.json").exists():
        raise FileExistsError("V3 recipe already frozen")
    protocol = Path("docs/RESEARCH_V3_PROTOCOL.md")
    recorded = root / "protocol.json"
    current = {
        "protocol_sha256": file_hash(protocol),
        "sample_sha256": file_hash(root / "sample.json"),
        "source_manifest_sha256": file_hash(Path("data/yambda/source_manifest.json")),
        "train": TRAIN,
        "search": SEARCH,
        "confirm": CONFIRM,
        "test": TEST,
        "seeds": SEEDS,
        "primary": ["global", "conditional"],
        "test_targets_constructed": False,
    }
    if recorded.exists():
        if load_json(recorded)["protocol_sha256"] != current["protocol_sha256"]:
            raise ValueError("Protocol changed after experiment start")
    else:
        write_json(recorded, current)
    write_json(root / "environment.json", environment())
    events, artists = load_music(root)
    resources = (
        load_json(root / "development_resources.json")
        if (root / "development_resources.json").exists()
        else []
    )
    frames = {}
    for day in (*TRAIN, SEARCH, CONFIRM):
        with measurement(resources, f"prepare/day{day}"):
            frames[day] = prepare_music(events, artists, day, root, targets=True)
        write_json(root / "development_resources.json", resources)
    manual = load_json(root / "cache/groups.json")["manual"]
    train = with_encoder_seed(pd.concat([frames[day] for day in TRAIN], ignore_index=True), 17)
    search = with_encoder_seed(frames[SEARCH], 17)
    truth = pd.read_parquet(root / "cache" / f"day{SEARCH}_truth.parquet")
    trials, recipe = [], {}
    for method in ("manual", "raw", "signed", "global", "conditional", "all"):
        columns = columns_for(method, manual)
        options = []
        for depth in (5, 7):
            with measurement(resources, f"search/{method}/{depth}"):
                model = fit_model(train, columns, depth, 600, 17)
                best = {"ndcg@10": -1.0}
                trajectory = []
                for index, scores in enumerate(
                    model.staged_predict(prediction_pool(search, columns), eval_period=25)
                ):
                    value = float(metric(search, scores, truth).ndcg.mean())
                    trajectory.append({"iterations": (index + 1) * 25, "ndcg@10": value})
                    if value > best["ndcg@10"]:
                        best = {"ndcg@10": value, "iterations": (index + 1) * 25}
                row = {
                    "method": method,
                    "depth": depth,
                    **best,
                    "trajectory": trajectory,
                    "features": len(columns),
                }
                options.append(row)
                trials.append(row)
            write_json(root / "search_trials.json", trials)
            write_json(root / "development_resources.json", resources)
            print("SEARCH", {k: v for k, v in row.items() if k != "trajectory"}, flush=True)
        recipe[method] = {**max(options, key=lambda r: r["ndcg@10"]), "columns": columns}
    recipe["selected_by_search"] = max(recipe, key=lambda m: recipe[m]["ndcg@10"])
    recipe["primary"] = {"first": "global", "second": "conditional"}
    recipe["protocol_sha256"] = file_hash(recorded)
    recipe["external_labels_used"] = False
    write_json(root / "frozen_recipe.json", recipe)
    print("FROZEN", recipe["selected_by_search"], flush=True)
    training = pd.concat([frames[day] for day in (*TRAIN, SEARCH)], ignore_index=True)
    evaluate(
        training, {CONFIRM: frames[CONFIRM]}, recipe, root / "confirm", (17,), root, resources, True
    )
    write_json(root / "development_resources.json", resources)


def evaluate(training, frames, recipe, output, seeds, root, resources, evaluate_targets):
    output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        train = with_encoder_seed(training, seed)
        for method in ("manual", "raw", "signed", "global", "conditional", "all"):
            spec = recipe[method]
            with measurement(resources, f"{output.name}/{method}/{seed}"):
                model = fit_model(train, spec["columns"], spec["depth"], spec["iterations"], seed)
                model.save_model(str(output / f"{method}_seed{seed}.cbm"))
                for day, original in frames.items():
                    frame = with_encoder_seed(original, seed)
                    values = score(model, frame, spec["columns"])
                    frame[["user_id", "item_id"]].assign(score=values).to_parquet(
                        output / f"{method}_seed{seed}_day{day}_predictions.parquet", index=False
                    )
        write_json(root / f"{output.name}_resources.json", resources)
    for day, frame in frames.items():
        for method, column in (
            ("popularity", "pop_score"),
            ("repeat", "repeat_score"),
            ("userknn", "knn_score"),
        ):
            frame[["user_id", "item_id"]].assign(score=frame[column]).to_parquet(
                output / f"{method}_seed0_day{day}_predictions.parquet", index=False
            )
    if evaluate_targets:
        truths = {
            day: pd.read_parquet(root / "cache" / f"day{day}_truth.parquet") for day in frames
        }
        summarize_output(output, truths, recipe)


def summarize_output(output, truths, recipe):
    records, tables = [], {}
    for path in sorted(output.glob("*_predictions.parquet")):
        name, suffix = path.stem.removesuffix("_predictions").rsplit("_seed", 1)
        seed, day = map(int, suffix.split("_day"))
        truth = truths[day]
        frame = pd.read_parquet(path)
        # All warm users were scored before labels; only measurable user queries enter metrics.
        frame = frame[frame.user_id.isin(truth.user_id)].reset_index(drop=True)
        per_user = metric(frame, frame.score.to_numpy(), truth)
        per_user["day"] = day
        per_user.to_parquet(output / f"{name}_seed{seed}_day{day}_users.parquet", index=False)
        tables.setdefault(name, []).append(per_user)
        records.append({"method": name, "seed": seed, "day": day, **summarize(per_user)})
    means = {}
    for name, blocks in tables.items():
        user_day = (
            pd.concat(blocks).groupby(["user_id", "day"], as_index=False).mean(numeric_only=True)
        )
        means[name] = (
            user_day.drop(columns="day").groupby("user_id", as_index=False).mean(numeric_only=True)
        )
        user_day.to_parquet(output / f"{name}_user_days.parquet", index=False)
        means[name].to_parquet(output / f"{name}_users.parquet", index=False)
    comparisons = {
        name: paired_bootstrap(means["manual"], data, 2028)
        for name, data in means.items()
        if name != "manual"
    }
    primary = paired_bootstrap(means["global"], means["conditional"], 2028)
    write_json(
        output / "results.json",
        {
            "metrics": records,
            "comparisons_vs_manual": comparisons,
            "primary_conditional_minus_global": primary,
            "selected_by_search": recipe["selected_by_search"],
            "users": len(means["manual"]),
            "days": sorted(truths),
        },
    )
    print(
        output,
        {name: float(data.ndcg.mean()) for name, data in means.items()},
        "PRIMARY",
        primary,
        flush=True,
    )


def test(root):
    recipe = load_json(root / "frozen_recipe.json")
    recipe_hash = file_hash(root / "frozen_recipe.json")
    if recipe["external_labels_used"] or recipe["protocol_sha256"] != file_hash(
        root / "protocol.json"
    ):
        raise ValueError("Invalid frozen recipe")
    if (root / "test/complete.json").exists():
        raise FileExistsError("V3 external test already opened")
    events, artists = load_music(root)
    resources = []
    frames = {}
    for day in TEST:
        with measurement(resources, f"test/prepare/day{day}"):
            frames[day] = prepare_music(events, artists, day, root, targets=False)
        write_json(root / "test_resources.json", resources)
    training = pd.concat(
        [
            pd.read_parquet(root / "cache" / f"day{day}.parquet")
            for day in (*TRAIN, SEARCH, CONFIRM)
        ],
        ignore_index=True,
    )
    evaluate(training, frames, recipe, root / "test", SEEDS, root, resources, False)
    if file_hash(root / "frozen_recipe.json") != recipe_hash:
        raise ValueError("Recipe changed during inference")
    # First construction of the held-out positive item sets, after all prediction arrays exist.
    truths = {}
    for day, frame in frames.items():
        truth = music_truth(events, day)
        truth = truth[truth.user_id.isin(frame.user_id)]
        truth.to_parquet(root / "test" / f"day{day}_truth.parquet", index=False)
        truths[day] = truth
    summarize_output(root / "test", truths, recipe)
    write_json(root / "test_resources.json", resources)
    write_json(
        root / "test/complete.json",
        {"recipe_sha256": recipe_hash, "results_sha256": file_hash(root / "test/results.json")},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["development", "test"])
    parser.add_argument("--root", type=Path, default=Path("runs/v3"))
    args = parser.parse_args()
    {"development": development, "test": test}[args.phase](args.root)

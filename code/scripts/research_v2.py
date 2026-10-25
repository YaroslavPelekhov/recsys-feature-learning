"""Second research stage. Run development, then external, then report.

The external command requires a frozen recipe. It scores the entire warm reserved
cohort before loading its future targets for evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from rec.data import sample_users
from rec.candidate_ablation import score_original_pool
from rec.experiment import fit_model, measurement, score_model, tune_trial
from rec.features import FeatureBuilder
from rec.io import environment, file_hash, write_json
from rec.linear import HistoricalMatrix
from rec.metrics import paired_bootstrap, summarize, user_metrics
from rec.protocol import Config, Fold, future_truth, history_before
from rec.retrieval import retrieve
from rec.routing import TemporalRouter

ROOT = Path("runs/v2")
DEV_RUN = Path("reports/main/evidence")
REPLICATION_RUN = Path("reports/replication/evidence")
CONFIG = Config(iterations=600, search_windows=(), search_half_lives=(), threads=12)
DEV_DATES = (
    "2021-06-28",
    "2021-07-05",
    "2021-07-12",
    "2021-07-19",
    "2021-07-26",
    "2021-08-02",
    "2021-08-09",
    "2021-08-16",
)
EXTERNAL_DATES = (
    "2021-05-17",
    "2021-05-24",
    "2021-05-31",
    "2021-06-07",
    "2021-06-14",
    "2021-06-21",
)


def reserve():
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / "reserved_users.json"
    if target.exists():
        return
    used = set()
    for previous in (DEV_RUN, REPLICATION_RUN):
        used.update(json.loads((previous / "sample.json").read_text("utf-8"))["sampled_user_ids"])
    with zipfile.ZipFile("data/raw/data_original.zip") as archive:
        with archive.open("data_original/interactions.csv") as stream:
            population = pd.read_csv(stream, usecols=["user_id"]).user_id.unique()
    available = population[~np.isin(population, list(used))]
    chosen = sample_users(available, 10000, 314159)
    write_json(
        target,
        {
            "seed": 314159,
            "n_users": len(chosen),
            "user_ids": chosen.tolist(),
            "excluded_previous_users": len(used),
            "overlap_with_development": len(set(chosen) & used),
            "target_cutoff": "2021-06-21",
            "target_end_exclusive": "2021-06-28",
        },
    )
    write_json(
        ROOT / "protocol.json",
        {
            "config": CONFIG.to_dict(),
            "dev_dates": DEV_DATES,
            "external_dates": EXTERNAL_DATES,
            "external_user_manifest_sha256": file_hash(target),
            "protocol_sha256": file_hash(Path("docs/RESEARCH_V2_PROTOCOL.md")),
        },
    )


def load_population(user_ids):
    with zipfile.ZipFile("data/raw/data_original.zip") as archive:
        with archive.open("data_original/interactions.csv") as stream:
            events = pd.read_csv(stream, usecols=["user_id", "item_id", "last_watch_dt"])
        events = events[events.user_id.isin(user_ids)].rename(
            columns={"last_watch_dt": "timestamp"}
        )
        events["timestamp"] = pd.to_datetime(events.timestamp)
        with archive.open("data_original/items.csv") as stream:
            items = pd.read_csv(stream).set_index("item_id")
        with archive.open("data_original/users.csv") as stream:
            users = pd.read_csv(stream).set_index("user_id").reindex(user_ids)
    return (
        events.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True),
        items,
        users,
    )


def prepare(events, target_events, items, users, cutoff_string, output, query_ids=None):
    cutoff = pd.Timestamp(cutoff_string)
    key = cutoff.strftime("%Y-%m-%d")
    cached = output / f"{key}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    output.mkdir(parents=True, exist_ok=True)
    history = history_before(events, cutoff)
    matrix = HistoricalMatrix.build(history, cutoff)
    fold = Fold(key, cutoff, cutoff + pd.Timedelta(days=7))
    if query_ids is None:
        truth = future_truth(target_events, fold, history)
        truth = truth[truth.user_id.isin(matrix.users)]
        queries = np.sort(truth.user_id.unique())
    else:
        queries = np.intersect1d(query_ids, matrix.users)
        truth = None
    learned = {}
    coefficients = matrix.ease(1000, threads=12)
    additions = matrix.recommend(coefficients, queries, 30, 120)
    retrieval = retrieve(history, items, cutoff, CONFIG, query_users=queries, extra_items=additions)
    pairs = retrieval.candidates
    for regularization in (1000, 100):
        if regularization == 100:
            coefficients = matrix.ease(100, threads=12)
        for half_life in (None, 7, 30):
            name = f"ease_l{regularization}_h{half_life}"
            learned[name] = matrix.pair_scores(pairs, coefficients, half_life)
    frame, groups = FeatureBuilder(history, items, users, cutoff).build(pairs, CONFIG)
    for name, values in learned.items():
        frame[name] = values
    frame["expert_zero"] = 0.0
    if truth is not None:
        frame = frame.merge(
            truth.assign(label=np.int8(1)),
            on=["user_id", "item_id"],
            how="left",
            validate="one_to_one",
        )
        frame["label"] = frame.label.fillna(0).astype(np.int8)
        truth.to_parquet(output / f"{key}_truth.parquet", index=False)
    else:
        frame["label"] = np.int8(0)  # Placeholder is never supplied to prediction_pool.
    frame["query_id"] = key + ":" + frame.user_id.astype(str)
    frame.to_parquet(cached, index=False)
    write_json(output / "groups.json", groups)
    write_json(
        output / f"{key}_audit.json",
        {
            "cutoff": cutoff_string,
            "history_rows": len(history),
            "catalog": len(matrix.items),
            "users": len(queries),
            "candidates": len(frame),
            "max_candidates": int(frame.groupby("user_id").size().max()),
            "future_labels_loaded": truth is not None,
            "history_max_timestamp": history.timestamp.max(),
        },
    )
    return frame


def truth_for(cache, date):
    return pd.read_parquet(cache / f"{date}_truth.parquet")


def metric(frame, scores, truth):
    return summarize(user_metrics(frame, scores, truth, np.sort(truth.user_id.unique())))


def oof_baseline(frames, columns, depth, iterations, cache):
    scored = []
    for index in range(1, len(frames)):
        frame = frames[index].copy()
        destination = cache / f"oof_{index}.parquet"
        if destination.exists():
            stored = pd.read_parquet(destination)
            if not np.array_equal(frame[["user_id", "item_id"]], stored[["user_id", "item_id"]]):
                raise ValueError("OOF cache alignment mismatch")
            frame["base_score"] = stored.base_score.to_numpy()
        else:
            prefix = pd.concat(frames[:index], ignore_index=True)
            model = fit_model(prefix, columns, CONFIG, depth, 17, iterations)
            frame["base_score"] = score_model(model, frame, columns)
            frame[["user_id", "item_id", "base_score"]].to_parquet(destination, index=False)
        scored.append(frame)
    return pd.concat(scored, ignore_index=True)


def development():
    reserve()
    if (ROOT / "frozen_recipe.json").exists():
        raise FileExistsError("Development recipe already frozen")
    write_json(ROOT / "environment.json", environment())
    ids = json.loads((DEV_RUN / "sample.json").read_text("utf-8"))["sampled_user_ids"]
    events, items, users = load_population(ids)
    cache = ROOT / "development"
    resources = []
    frames = {}
    for date in DEV_DATES[:-1]:
        with measurement(resources, f"prepare/{date}"):
            frames[date] = prepare(events, events, items, users, date, cache)
        write_json(ROOT / "development_resources.json", resources)
    groups = json.loads((cache / "groups.json").read_text("utf-8"))
    manual = groups["manual"]
    learned = [name for name in frames[DEV_DATES[0]] if name.startswith("ease_")]
    experts = learned + ["retrieval_pop", "retrieval_cf", "retrieval_content", "expert_zero"]
    training = [frames[date] for date in DEV_DATES[:5]]
    train = pd.concat(training, ignore_index=True)
    search = frames["2021-08-02"].copy()
    search_truth = truth_for(cache, "2021-08-02")
    trials = []
    recipe = {}
    for method, columns in (("manual", manual), ("hybrid", manual + learned)):
        options = []
        for depth in (5, 7):
            with measurement(resources, f"search/{method}/{depth}"):
                row = {
                    "method": method,
                    **tune_trial(train, search, search_truth, columns, CONFIG, depth),
                }
            trials.append(row)
            options.append(row)
            write_json(ROOT / "search_trials.json", trials)
        recipe[method] = {**max(options, key=lambda r: r["ndcg"]), "columns": columns}
    baseline = recipe["manual"]
    with measurement(resources, "out_of_time_teacher"):
        oof = oof_baseline(training, manual, baseline["depth"], baseline["iterations"], cache)
        model = fit_model(train, manual, CONFIG, baseline["depth"], 17, baseline["iterations"])
        search["base_score"] = score_model(model, search, manual)
    for method, adaptive, offset, scales in (
        ("global_mix", False, None, (1.0,)),
        ("adaptive_mix", True, None, (1.0,)),
        ("residual_router", True, "base_score", (0.25, 0.5, 1.0)),
    ):
        options = []
        for l2 in (0.01, 0.1):
            for scale in scales:
                start = time.perf_counter()
                router = TemporalRouter(
                    experts, adaptive=adaptive, l2=l2, residual_scale=scale, offset_column=offset
                ).fit(oof)
                scores = router.predict(search)
                row = {
                    "method": method,
                    "l2": l2,
                    "scale": scale,
                    "ndcg": metric(search, scores, search_truth)["ndcg@10"],
                    "seconds": time.perf_counter() - start,
                    "adaptive": adaptive,
                    "offset_column": offset,
                    "optimizer": router.optimization_,
                }
                trials.append(row)
                options.append(row)
                write_json(ROOT / "search_trials.json", trials)
                print(row, flush=True)
        recipe[method] = {**max(options, key=lambda r: r["ndcg"]), "experts": experts}
    # Single primary comparison fixed by development search; report all predeclared methods.
    candidates = ("hybrid", "global_mix", "adaptive_mix", "residual_router")
    recipe["selected"] = max(candidates, key=lambda name: recipe[name]["ndcg"])
    recipe["external_labels_used"] = False
    recipe["development_dates"] = DEV_DATES
    recipe["reserved_users_sha256"] = file_hash(ROOT / "reserved_users.json")
    write_json(ROOT / "frozen_recipe.json", recipe)
    write_json(ROOT / "development_resources.json", resources)
    print(
        "FROZEN",
        recipe["selected"],
        {name: recipe[name]["ndcg"] for name in ("manual", *candidates)},
        flush=True,
    )
    evaluate_block(
        training + [search],
        frames["2021-08-09"],
        truth_for(cache, "2021-08-09"),
        recipe,
        cache,
        ROOT / "confirm",
        (17,),
        resources,
    )
    # The old August test is explicitly exploratory and cannot select the frozen recipe.
    test = prepare(events, events, items, users, "2021-08-16", cache)
    evaluate_block(
        training + [search, frames["2021-08-09"]],
        test,
        truth_for(cache, "2021-08-16"),
        recipe,
        cache,
        ROOT / "exploratory_test",
        (17, 42, 73),
        resources,
    )
    write_json(ROOT / "development_resources.json", resources)


def fit_router(specification, data):
    return TemporalRouter(
        specification["experts"],
        adaptive=specification["adaptive"],
        l2=specification["l2"],
        residual_scale=specification["scale"],
        offset_column=specification["offset_column"],
    ).fit(data)


def evaluate_block(training, frame, truth, recipe, cache, output, seeds, resources, evaluate=True):
    output.mkdir(parents=True, exist_ok=True)
    train = pd.concat(training, ignore_index=True)
    manual = recipe["manual"]
    with measurement(resources, f"{output.name}/oof"):
        oof = oof_baseline(
            training, manual["columns"], manual["depth"], manual["iterations"], cache
        )
    routers = {
        name: fit_router(recipe[name], oof)
        for name in ("global_mix", "adaptive_mix", "residual_router")
    }
    for name, model in routers.items():
        write_json(output / f"{name}_model.json", model.state())
    all_rows = []
    per_seed = {}
    for seed in seeds:
        scored = frame.copy()
        for name in ("manual", "hybrid"):
            specification = recipe[name]
            with measurement(resources, f"{output.name}/{name}/{seed}"):
                model = fit_model(
                    train,
                    specification["columns"],
                    CONFIG,
                    specification["depth"],
                    seed,
                    specification["iterations"],
                )
                scores = score_model(model, scored, specification["columns"])
                model.save_model(str(output / f"{name}_seed{seed}.cbm"))
            if name == "manual":
                scored["base_score"] = scores
            per_seed[name, seed] = scores
        for name, model in routers.items():
            per_seed[name, seed] = model.predict(scored)
    per_seed["ease", 0] = frame["ease_l1000_h30"].to_numpy()
    per_seed["popularity", 0] = frame.retrieval_pop.to_numpy()
    # Persist every prediction before the caller loads external labels.
    for (name, seed), scores in per_seed.items():
        frame[["user_id", "item_id"]].assign(score=scores).to_parquet(
            output / f"{name}_seed{seed}_predictions.parquet", index=False
        )
    if not evaluate:
        return per_seed
    users = np.sort(truth.user_id.unique())
    by_method = {}
    for (name, seed), scores in per_seed.items():
        per_user = user_metrics(frame, scores, truth, users)
        by_method.setdefault(name, []).append(per_user)
        all_rows.append({"method": name, "seed": seed, **summarize(per_user)})
        per_user.to_parquet(output / f"{name}_seed{seed}_users.parquet", index=False)
    means = {}
    for name, tables in by_method.items():
        means[name] = pd.concat(tables).groupby("user_id", as_index=False).mean(numeric_only=True)
        means[name].to_parquet(output / f"{name}_users.parquet", index=False)
    comparisons = {
        name: paired_bootstrap(means["manual"], table, 314159)
        for name, table in means.items()
        if name != "manual"
    }
    write_json(
        output / "results.json",
        {"metrics": all_rows, "comparisons": comparisons, "primary": recipe["selected"]},
    )
    print(output, {name: table.ndcg.mean() for name, table in means.items()}, flush=True)
    return per_seed


def external():
    frozen = ROOT / "frozen_recipe.json"
    recipe = json.loads(frozen.read_text("utf-8"))
    frozen_hash = file_hash(frozen)
    reserve_data = json.loads((ROOT / "reserved_users.json").read_text("utf-8"))
    if file_hash(ROOT / "reserved_users.json") != recipe["reserved_users_sha256"]:
        raise ValueError("Reserved cohort changed after recipe freeze")
    if (ROOT / "external/complete.json").exists():
        raise FileExistsError("External test already evaluated")
    dev_ids = json.loads((DEV_RUN / "sample.json").read_text("utf-8"))["sampled_user_ids"]
    held_ids = np.array(reserve_data["user_ids"])
    events, items, users = load_population(list(dev_ids) + held_ids.tolist())
    dev_targets = events[events.user_id.isin(dev_ids)]
    cache = ROOT / "external_cache"
    resources = []
    frames = []
    for date in EXTERNAL_DATES[:-1]:
        with measurement(resources, f"external/prepare/{date}"):
            frames.append(prepare(events, dev_targets, items, users, date, cache))
    with measurement(resources, "external/prepare/test_without_labels"):
        test = prepare(events, None, items, users, EXTERNAL_DATES[-1], cache, query_ids=held_ids)
    # No held-out target is constructed before all methods and prediction arrays are fixed.
    output = ROOT / "external"
    evaluate_block(
        frames, test, None, recipe, cache, output, (17, 42, 73), resources, evaluate=False
    )
    with measurement(resources, "external/original_candidate_policy_control"):
        score_original_pool(frames, test, recipe, CONFIG, output / "candidate_ablation")
    if file_hash(frozen) != frozen_hash:
        raise ValueError("Recipe changed after freeze")
    cutoff = pd.Timestamp(EXTERNAL_DATES[-1])
    history = history_before(events, cutoff)
    external_events = events[events.user_id.isin(held_ids)]
    truth = future_truth(
        external_events, Fold("external", cutoff, cutoff + pd.Timedelta(days=7)), history
    )
    truth = truth[truth.user_id.isin(test.user_id)]
    truth.to_parquet(output / "truth.parquet", index=False)
    metrics, tables = [], {}
    for path in sorted(output.rglob("*_predictions.parquet")):
        name, seed = path.stem.removesuffix("_predictions").rsplit("_seed", 1)
        predictions = pd.read_parquet(path)
        predictions = predictions[predictions.user_id.isin(truth.user_id)].reset_index(drop=True)
        per_user = user_metrics(
            predictions, predictions.score.to_numpy(), truth, np.sort(truth.user_id.unique())
        )
        tables.setdefault(name, []).append(per_user)
        metrics.append({"method": name, "seed": int(seed), **summarize(per_user)})
    means = {}
    for name, blocks in tables.items():
        means[name] = pd.concat(blocks).groupby("user_id", as_index=False).mean(numeric_only=True)
        means[name].to_parquet(output / f"{name}_users.parquet", index=False)
    comparisons = {
        name: paired_bootstrap(means["manual"], block, 314159)
        for name, block in means.items()
        if name != "manual"
    }
    write_json(
        output / "results.json",
        {
            "metrics": metrics,
            "comparisons": comparisons,
            "primary": recipe["selected"],
            "frozen_recipe_sha256": frozen_hash,
            "external_period": ["2021-06-21", "2021-06-28"],
            "target_users": truth.user_id.nunique(),
            "target_rows": len(truth),
        },
    )
    write_json(ROOT / "external_resources.json", resources)
    write_json(
        output / "complete.json",
        {"results_sha256": file_hash(output / "results.json"), "frozen_recipe_sha256": frozen_hash},
    )
    print(
        "EXTERNAL",
        {name: block.ndcg.mean() for name, block in means.items()},
        comparisons,
        flush=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["reserve", "development", "external"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--development-run", type=Path, default=DEV_RUN)
    parser.add_argument("--replication-run", type=Path, default=REPLICATION_RUN)
    args = parser.parse_args()
    ROOT, DEV_RUN, REPLICATION_RUN = args.root, args.development_run, args.replication_run
    {"reserve": reserve, "development": development, "external": external}[args.phase]()

"""Reserve, tune and evaluate the fixed reconstruction factorial on a fresh cohort."""

import argparse
import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRanker

import research_v3 as study
from parallel_v3 import model_frame
from rec.data import sample_users
from rec.experiment import measurement
from rec.io import environment, file_hash, write_json
from rec.metrics import paired_bootstrap, summarize, user_metrics
from rec.music import load_music, music_truth, prepare_music
from rec.reconstruction import VARIANTS, reconstructed_scores
from rec.transition import DAY, TransitionEmbedding
from support_v3 import fuse

SEED = 424242
CODE = (
    "src/rec/reconstruction.py",
    "src/rec/transition.py",
    "src/rec/music.py",
    "src/rec/metrics.py",
    "scripts/research_v4.py",
    "scripts/research_v3.py",
    "scripts/parallel_v3.py",
    "scripts/support_v3.py",
    "pyproject.toml",
    "uv.lock",
)


def reserve(root):
    if (root / "sample.json").exists():
        raise FileExistsError("V4 cohort already reserved")
    source = Path("data/yambda/listens.parquet")
    manifest = study.load_json(Path("data/yambda/source_manifest.json"))
    if file_hash(source) != manifest["files"]["listens.parquet"]["sha256"]:
        raise ValueError("Pinned listens changed")
    previous = study.load_json(Path("reports/v3/evidence/sample.json"))
    original = pq.ParquetFile(source)
    population = np.unique(
        np.concatenate(
            [
                original.read_row_group(i, columns=["uid"])["uid"].unique().to_numpy()
                for i in range(original.num_row_groups)
            ]
        )
    )
    eligible = np.setdiff1d(population, previous["sampled_user_ids"])
    selected = sample_users(eligible, 1000, SEED)
    events = pd.read_parquet(source, filters=[("uid", "in", selected.tolist())])
    root.mkdir(parents=True, exist_ok=True)
    events.to_parquet(root / "sampled_events.parquet", index=False)
    write_json(
        root / "sample.json",
        {
            "seed": SEED,
            "n_users": len(selected),
            "rows": len(events),
            "sampled_user_ids": selected.tolist(),
            "source_users": len(population),
            "eligible_users": len(eligible),
            "overlap_previous": 0,
            "excluded_sample_sha256": file_hash(Path("reports/v3/evidence/sample.json")),
            "all_selected_source_rows_retained": True,
        },
    )
    write_json(
        root / "protocol.json",
        {
            "document_sha256": file_hash(Path("docs/RESEARCH_V4_PROTOCOL.md")),
            "sample_sha256": file_hash(root / "sample.json"),
            "events_sha256": file_hash(root / "sampled_events.parquet"),
            "source_manifest_sha256": file_hash(Path("data/yambda/source_manifest.json")),
            "primary": ["G0", "A1"],
            "test_targets_constructed": False,
            "test": study.TEST,
            "seeds": study.SEEDS,
            "half_life_hours": 24,
            "external_kind": "disjoint user cohort, same anonymous period",
        },
    )
    write_json(root / "environment.json", environment())
    print("RESERVED", len(events), "events; zero UID overlap", flush=True)


def score_day(base, root, day, seed, events, directory="scores"):
    path = root / directory / f"day{day}_seed{seed}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame = pd.read_parquet(base / "cache" / f"day{day}.parquet", columns=["user_id", "item_id"])
    encoder = TransitionEmbedding.load(base / "encoders" / f"day{day}" / f"global_seed{seed}.npz")
    cutoff = day * DAY - 1800
    values, audit = reconstructed_scores(frame, events[events.timestamp < cutoff], encoder, cutoff)
    path.parent.mkdir(parents=True, exist_ok=True)
    values.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), audit)
    print("SCORED", day, seed, audit, flush=True)
    return values


def development(root, base):
    if (root / "frozen_recipe.json").exists():
        raise FileExistsError("V4 already frozen")
    protocol = study.load_json(root / "protocol.json")
    if protocol["document_sha256"] != file_hash(Path("docs/RESEARCH_V4_PROTOCOL.md")):
        raise ValueError("Protocol changed after reservation")
    if any((root / "test" / f"day{day}_truth.parquet").exists() for day in study.TEST):
        raise ValueError("New cohort test already open")
    spec = study.load_json(Path("reports/v3/evidence/frozen_recipe.json"))["manual"]
    events, _ = load_music(base)
    scores = score_day(base, root, study.SEARCH, 17, events, "development/scores")
    destination = root / "development/manual.cbm"
    destination.parent.mkdir(parents=True, exist_ok=True)
    old = Path("runs/v3_support/manual_search.cbm")
    if not destination.exists():
        if old.exists():
            os.link(old.resolve(), destination.resolve())
        else:
            train = pd.concat(
                [model_frame(base, d, spec["columns"], 17) for d in study.TRAIN], ignore_index=True
            )
            model = study.fit_model(train, spec["columns"], spec["depth"], spec["iterations"], 17)
            model.save_model(str(destination))
    model = CatBoostRanker().load_model(str(destination))
    frame = model_frame(base, study.SEARCH, spec["columns"], 17)
    if model.feature_names_ != spec["columns"] or not np.array_equal(
        frame[["user_id", "item_id"]], scores[["user_id", "item_id"]]
    ):
        raise ValueError("Development baseline alignment mismatch")
    manual = study.score(model, frame, spec["columns"])
    truth = pd.read_parquet(base / "cache/day294_truth.parquet")
    users = np.sort(truth.user_id.unique())
    trials, standalone = [], []
    for name in (*VARIANTS, "legacy"):
        standalone.append(
            {
                "method": name,
                **summarize(user_metrics(scores, scores[name].to_numpy(), truth, users)),
            }
        )
        if name == "legacy":
            continue
        for alpha in (0.0, 0.05, 0.1, 0.25, 0.5):
            values = fuse(scores, manual, scores[name].to_numpy(), alpha)
            trials.append(
                {
                    "method": name,
                    "alpha": alpha,
                    **summarize(user_metrics(scores, values, truth, users)),
                }
            )
    write_json(root / "search_trials.json", trials)
    write_json(root / "search_standalone.json", standalone)
    recipe = {
        "manual": spec,
        "fusion": {
            name: max((r for r in trials if r["method"] == name), key=lambda r: r["ndcg@10"])
            for name in VARIANTS
        },
        "selected_fusion": max(trials, key=lambda r: r["ndcg@10"]),
        "primary": ["G0", "A1"],
        "half_life_hours": 24,
        "protocol_sha256": file_hash(root / "protocol.json"),
        "external_labels_used": False,
        "development_manual_sha256": file_hash(destination),
        "development_scores_sha256": file_hash(root / "development/scores/day294_seed17.parquet"),
        "code": {name: file_hash(Path(name)) for name in CODE},
    }
    write_json(root / "frozen_recipe.json", recipe)
    print("FROZEN", recipe["selected_fusion"], flush=True)
    print("STANDALONE", {r["method"]: r["ndcg@10"] for r in standalone}, flush=True)


def manual_job(root, seed):
    recipe = study.load_json(root / "frozen_recipe.json")
    spec = recipe["manual"]
    output = root / "test"
    job = output / f"manual_seed{seed}_job.json"
    if job.exists():
        saved = study.load_json(job)
        for name, digest in saved["outputs"].items():
            if file_hash(output / name) != digest:
                raise ValueError("Cached baseline job changed")
        return saved["resources"]
    resources = []
    with measurement(resources, f"manual/{seed}"):
        train = pd.concat(
            [
                model_frame(root, d, spec["columns"], seed)
                for d in (*study.TRAIN, study.SEARCH, study.CONFIRM)
            ],
            ignore_index=True,
        )
        model = study.fit_model(train, spec["columns"], spec["depth"], spec["iterations"], seed)
        model.save_model(str(output / f"manual_seed{seed}.cbm"))
        for day in study.TEST:
            frame = model_frame(root, day, spec["columns"], seed)
            values = study.score(model, frame, spec["columns"])
            frame[["user_id", "item_id"]].assign(score=values).to_parquet(
                output / f"manual_seed{seed}_day{day}_predictions.parquet", index=False
            )
    paths = [
        output / f"manual_seed{seed}.cbm",
        *[output / f"manual_seed{seed}_day{d}_predictions.parquet" for d in study.TEST],
    ]
    write_json(job, {"outputs": {p.name: file_hash(p) for p in paths}, "resources": resources})
    return resources


def validate_freeze(root):
    recipe = study.load_json(root / "frozen_recipe.json")
    if recipe["external_labels_used"] or recipe["protocol_sha256"] != file_hash(
        root / "protocol.json"
    ):
        raise ValueError("Frozen recipe changed")
    for name, digest in recipe["code"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError(f"Frozen implementation changed: {name}")
    protocol = study.load_json(root / "protocol.json")
    for path, key in (
        (root / "sample.json", "sample_sha256"),
        (root / "sampled_events.parquet", "events_sha256"),
        (Path("docs/RESEARCH_V4_PROTOCOL.md"), "document_sha256"),
    ):
        if file_hash(path) != protocol[key]:
            raise ValueError(f"Frozen input changed: {path}")
    return recipe


def external(root):
    recipe = validate_freeze(root)
    if (root / "test/complete.json").exists():
        raise FileExistsError("V4 external test already opened")
    if any((root / "test" / f"day{d}_truth.parquet").exists() for d in study.TEST):
        raise ValueError("Incomplete run already has targets; do not resume inference")
    resources = (
        study.load_json(root / "resources.json") if (root / "resources.json").exists() else []
    )
    events, artists = load_music(root)
    for day in (*study.TRAIN, study.SEARCH, study.CONFIRM, *study.TEST):
        if not (root / "cache" / f"day{day}.parquet").exists():
            with measurement(resources, f"prepare/{day}"):
                prepare_music(events, artists, day, root, targets=day not in study.TEST)
            write_json(root / "resources.json", resources)
    del events, artists
    gc.collect()
    output = root / "test"
    output.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as executor:
        for job in as_completed([executor.submit(manual_job, root, seed) for seed in study.SEEDS]):
            costs = job.result()
            if costs[0]["stage"] not in {r["stage"] for r in resources}:
                resources.extend(costs)
            write_json(root / "resources.json", resources)
            print("FIT", costs, flush=True)
    events, _ = load_music(root)
    for day in study.TEST:
        for seed in study.SEEDS:
            with measurement(resources, f"reconstruction/{day}/{seed}"):
                scores = score_day(root, root, day, seed, events)
                manual = pd.read_parquet(output / f"manual_seed{seed}_day{day}_predictions.parquet")
                if not np.array_equal(
                    manual[["user_id", "item_id"]], scores[["user_id", "item_id"]]
                ):
                    raise ValueError("External score alignment mismatch")
                for name in (*VARIANTS, "legacy"):
                    scores[["user_id", "item_id"]].assign(score=scores[name]).to_parquet(
                        output / f"{name}_seed{seed}_day{day}_predictions.parquet", index=False
                    )
                for name in VARIANTS:
                    values = fuse(
                        scores,
                        manual.score.to_numpy(),
                        scores[name].to_numpy(),
                        recipe["fusion"][name]["alpha"],
                    )
                    scores[["user_id", "item_id"]].assign(score=values).to_parquet(
                        output / f"fusion{name}_seed{seed}_day{day}_predictions.parquet",
                        index=False,
                    )
            write_json(root / "resources.json", resources)
        frame = pd.read_parquet(root / "cache" / f"day{day}.parquet")
        for name, column in (
            ("repeat", "repeat_score"),
            ("popularity", "pop_score"),
            ("userknn", "knn_score"),
        ):
            frame[["user_id", "item_id"]].assign(score=frame[column]).to_parquet(
                output / f"{name}_seed0_day{day}_predictions.parquet", index=False
            )
    write_json(
        root / "predictions_ready.json",
        {
            "recipe_sha256": file_hash(root / "frozen_recipe.json"),
            "targets_exist": False,
            "predictions": {
                p.name: file_hash(p) for p in sorted(output.glob("*_predictions.parquet"))
            },
        },
    )
    # First construction of held-out item sets happens only after all 66 forecasts.
    truths = {}
    for day in study.TEST:
        users = pd.read_parquet(
            root / "cache" / f"day{day}.parquet", columns=["user_id"]
        ).user_id.unique()
        truth = music_truth(events, day)
        truths[day] = truth[truth.user_id.isin(users)]
        truths[day].to_parquet(output / f"day{day}_truth.parquet", index=False)
    aggregate(root)


def aggregate(root):
    output = root / "test"
    rows, blocks = [], {}
    for path in sorted(output.glob("*_predictions.parquet")):
        method, suffix = path.stem.removesuffix("_predictions").rsplit("_seed", 1)
        seed, day = map(int, suffix.split("_day"))
        predictions = pd.read_parquet(path)
        truth = pd.read_parquet(output / f"day{day}_truth.parquet")
        predictions = predictions[predictions.user_id.isin(truth.user_id)].reset_index(drop=True)
        metrics = user_metrics(
            predictions, predictions.score.to_numpy(), truth, np.sort(truth.user_id.unique())
        )
        metrics["day"] = day
        metrics.to_parquet(output / f"{method}_seed{seed}_day{day}_users.parquet", index=False)
        blocks.setdefault(method, []).append(metrics)
        rows.append({"method": method, "seed": seed, "day": day, **summarize(metrics)})
    means = {}
    for method, tables in blocks.items():
        user_day = (
            pd.concat(tables).groupby(["user_id", "day"], as_index=False).mean(numeric_only=True)
        )
        means[method] = (
            user_day.drop(columns="day").groupby("user_id", as_index=False).mean(numeric_only=True)
        )
        means[method].to_parquet(output / f"{method}_users.parquet", index=False)
    comparisons = [
        ("G0", "A1"),
        ("G0", "G1"),
        ("A0", "A1"),
        ("G0", "A0"),
        ("G1", "A1"),
        ("legacy", "G0"),
    ]
    result = {
        "metrics": rows,
        "primary": paired_bootstrap(means["G0"], means["A1"], SEED),
        "factorial": {
            f"{b}_minus_{a}": paired_bootstrap(means[a], means[b], SEED) for a, b in comparisons
        },
        "vs_manual": {
            name: paired_bootstrap(means["manual"], table, SEED)
            for name, table in means.items()
            if name != "manual"
        },
        "overall": {name: summarize(table) for name, table in means.items()},
    }
    write_json(output / "results.json", result)
    write_json(
        output / "complete.json",
        {
            "results_sha256": file_hash(output / "results.json"),
            "recipe_sha256": file_hash(root / "frozen_recipe.json"),
        },
    )
    print(
        "COMPLETE",
        result["primary"],
        {k: v["ndcg@10"] for k, v in result["overall"].items()},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["reserve", "development", "external"])
    parser.add_argument("--root", type=Path, default=Path("runs/v4"))
    parser.add_argument("--base", type=Path, default=Path("runs/v3"))
    args = parser.parse_args()
    if args.phase == "development":
        development(args.root, args.base)
    else:
        {"reserve": reserve, "external": external}[args.phase](args.root)

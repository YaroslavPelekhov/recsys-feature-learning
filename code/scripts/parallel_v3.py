"""Equivalent v3 recipe with two independent CPU fits at a time (12 threads each)."""

import argparse
import gc
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from filelock import FileLock, Timeout

import research_v3 as study
from rec.experiment import measurement
from rec.io import environment, file_hash, write_json
from rec.music import load_music, music_truth, prepare_music

METHODS = ("manual", "raw", "signed", "global", "conditional", "all")


@contextmanager
def cpu_slot(root):
    acquired = None
    while acquired is None:
        for number in (0, 1):
            candidate = FileLock(str(root / f"cpu_slot{number}.lock"))
            try:
                candidate.acquire(timeout=0)
                acquired = candidate
                break
            except Timeout:
                continue
        if acquired is None:
            time.sleep(0.2)
    try:
        yield
    finally:
        acquired.release()


def merge_resources(resources, costs):
    stages = {row["stage"] for row in resources}
    resources.extend(row for row in costs if row["stage"] not in stages)


def model_frame(root, day, columns, seed):
    sources = [
        c.replace("global_h", f"global_seed{seed}_h").replace(
            "conditional_h", f"conditional_seed{seed}_h"
        )
        for c in columns
    ]
    needed = list(dict.fromkeys(["user_id", "item_id", "query_id", "label", *sources]))
    frame = pd.read_parquet(root / "cache" / f"day{day}.parquet", columns=needed)
    return frame.rename(columns=dict(zip(sources, columns)))


def search_job(root, method, depth):
    columns = study.columns_for(method, study.load_json(root / "cache/groups.json")["manual"])
    records = []
    with cpu_slot(root), measurement(records, f"search/{method}/{depth}"):
        train = pd.concat(
            [model_frame(root, day, columns, 17) for day in study.TRAIN], ignore_index=True
        )
        search = model_frame(root, study.SEARCH, columns, 17)
        truth = pd.read_parquet(root / "cache" / f"day{study.SEARCH}_truth.parquet")
        model = study.fit_model(train, columns, depth, 600, 17)
        best, trajectory = {"ndcg@10": -1.0}, []
        for index, values in enumerate(
            model.staged_predict(study.prediction_pool(search, columns), eval_period=25)
        ):
            value = float(study.metric(search, values, truth).ndcg.mean())
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
    write_json(
        root / "search_jobs" / f"{method}_depth{depth}.json", {"trial": row, "resources": records}
    )
    return row, records


def final_job(root, training_days, days, recipe, output, method, seed):
    job_path = output / f"{method}_seed{seed}_job.json"
    recipe_hash = file_hash(root / "frozen_recipe.json")
    if job_path.exists():
        job = study.load_json(job_path)
        if job["recipe_sha256"] != recipe_hash:
            raise ValueError("Final job belongs to a different frozen recipe")
        for name, digest in job["outputs"].items():
            if file_hash(output / name) != digest:
                raise ValueError("Final job output changed")
        return job["resources"]
    spec, records, paths = recipe[method], [], []
    with cpu_slot(root), measurement(records, f"{output.name}/{method}/{seed}"):
        train = pd.concat(
            [model_frame(root, day, spec["columns"], seed) for day in training_days],
            ignore_index=True,
        )
        model = study.fit_model(train, spec["columns"], spec["depth"], spec["iterations"], seed)
        model_path = output / f"{method}_seed{seed}.cbm"
        model.save_model(str(model_path))
        paths.append(model_path)
        for day in days:
            frame = model_frame(root, day, spec["columns"], seed)
            values = study.score(model, frame, spec["columns"])
            prediction_path = output / f"{method}_seed{seed}_day{day}_predictions.parquet"
            frame[["user_id", "item_id"]].assign(score=values).to_parquet(
                prediction_path, index=False
            )
            paths.append(prediction_path)
    write_json(
        job_path,
        {
            "recipe_sha256": recipe_hash,
            "resources": records,
            "outputs": {p.name: file_hash(p) for p in paths},
        },
    )
    return records


def evaluate(root, training_days, frames, recipe, output, seeds, resources, targets):
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                final_job, root, training_days, tuple(frames), recipe, output, method, seed
            )
            for seed in seeds
            for method in METHODS
        ]
        for future in as_completed(futures):
            costs = future.result()
            merge_resources(resources, costs)
            write_json(root / f"{output.name}_resources.json", resources)
            print("FIT", costs[-1], flush=True)
    for day, frame in frames.items():
        for method, column in (
            ("popularity", "pop_score"),
            ("repeat", "repeat_score"),
            ("userknn", "knn_score"),
        ):
            frame[["user_id", "item_id"]].assign(score=frame[column]).to_parquet(
                output / f"{method}_seed0_day{day}_predictions.parquet", index=False
            )
    if targets:
        truths = {
            day: pd.read_parquet(root / "cache" / f"day{day}_truth.parquet") for day in frames
        }
        study.summarize_output(output, truths, recipe)


def development(root):
    if (root / "frozen_recipe.json").exists():
        raise FileExistsError("V3 recipe already frozen")
    expected = {
        "protocol_sha256": file_hash(Path("docs/RESEARCH_V3_PROTOCOL.md")),
        "sample_sha256": file_hash(root / "sample.json"),
        "source_manifest_sha256": file_hash(Path("data/yambda/source_manifest.json")),
        "train": study.TRAIN,
        "search": study.SEARCH,
        "confirm": study.CONFIRM,
        "test": study.TEST,
        "seeds": study.SEEDS,
        "primary": ["global", "conditional"],
        "test_targets_constructed": False,
    }
    if (root / "protocol.json").exists():
        original = study.load_json(root / "protocol.json")
        for key in ("protocol_sha256", "sample_sha256", "source_manifest_sha256"):
            if expected[key] != original[key]:
                raise ValueError(f"Experiment input changed: {key}")
    else:
        write_json(root / "protocol.json", expected)
    write_json(root / "environment.json", environment())
    resources = (
        study.load_json(root / "development_resources.json")
        if (root / "development_resources.json").exists()
        else []
    )
    events, artists = load_music(root)
    for day in (*study.TRAIN, study.SEARCH, study.CONFIRM):
        if not (root / "cache" / f"day{day}.parquet").exists():
            with measurement(resources, f"prepare/day{day}"):
                prepare_music(events, artists, day, root, True)
            write_json(root / "development_resources.json", resources)
    del events, artists
    gc.collect()
    trials = (
        study.load_json(root / "search_trials.json")
        if (root / "search_trials.json").exists()
        else []
    )
    completed = {(row["method"], row["depth"]) for row in trials}
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for method in METHODS:
            for depth in (5, 7):
                if (method, depth) in completed:
                    continue
                cached = root / "search_jobs" / f"{method}_depth{depth}.json"
                if cached.exists():
                    job = study.load_json(cached)
                    trials.append(job["trial"])
                    merge_resources(resources, job["resources"])
                else:
                    futures.append(executor.submit(search_job, root, method, depth))
        for future in as_completed(futures):
            row, costs = future.result()
            trials.append(row)
            merge_resources(resources, costs)
            write_json(root / "search_trials.json", trials)
            write_json(root / "development_resources.json", resources)
            print("SEARCH", {k: v for k, v in row.items() if k != "trajectory"}, flush=True)
    manual = study.load_json(root / "cache/groups.json")["manual"]
    recipe = {
        method: {
            **max((r for r in trials if r["method"] == method), key=lambda r: r["ndcg@10"]),
            "columns": study.columns_for(method, manual),
        }
        for method in METHODS
    }
    recipe["selected_by_search"] = max(recipe, key=lambda method: recipe[method]["ndcg@10"])
    recipe["primary"] = {"first": "global", "second": "conditional"}
    recipe["protocol_sha256"] = file_hash(root / "protocol.json")
    recipe["external_labels_used"] = False
    write_json(root / "frozen_recipe.json", recipe)
    print("FROZEN", recipe["selected_by_search"], flush=True)
    frame = pd.read_parquet(root / "cache" / f"day{study.CONFIRM}.parquet")
    evaluate(
        root,
        (*study.TRAIN, study.SEARCH),
        {study.CONFIRM: frame},
        recipe,
        root / "confirm",
        (17,),
        resources,
        True,
    )
    write_json(root / "development_resources.json", resources)


def confirm(root):
    recipe = study.load_json(root / "frozen_recipe.json")
    source = root / "confirm_resources.json"
    resources = (
        study.load_json(source)
        if source.exists()
        else study.load_json(root / "development_resources.json")
    )
    frame = pd.read_parquet(root / "cache" / f"day{study.CONFIRM}.parquet")
    evaluate(
        root,
        (*study.TRAIN, study.SEARCH),
        {study.CONFIRM: frame},
        recipe,
        root / "confirm",
        (17,),
        resources,
        True,
    )
    write_json(root / "development_resources.json", resources)


def test(root, support_root=None):
    recipe = study.load_json(root / "frozen_recipe.json")
    digest = file_hash(root / "frozen_recipe.json")
    if recipe["external_labels_used"] or recipe["protocol_sha256"] != file_hash(
        root / "protocol.json"
    ):
        raise ValueError("Invalid frozen recipe")
    if (root / "test/complete.json").exists():
        raise FileExistsError("V3 test already opened")
    seal = root / "frozen_inputs.json"
    if not seal.exists():
        code = [
            "src/rec/transition.py",
            "src/rec/music.py",
            "scripts/research_v3.py",
            "scripts/parallel_v3.py",
            "pyproject.toml",
            "uv.lock",
        ]
        inputs = [
            "sampled_events.parquet",
            *[f"cache/day{day}.parquet" for day in (*study.TRAIN, study.SEARCH, study.CONFIRM)],
        ]
        write_json(
            seal,
            {
                "recipe_sha256": digest,
                "code": {name: file_hash(Path(name)) for name in code},
                "inputs": {name: file_hash(root / name) for name in inputs},
            },
        )
    sealed = study.load_json(seal)
    if sealed["recipe_sha256"] != digest:
        raise ValueError("Sealed inputs belong to a different recipe")
    for name, expected in sealed["code"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Core code changed after test input freeze: {name}")
    for name, expected in sealed["inputs"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Training input changed after freeze: {name}")
    resources = (
        study.load_json(root / "test_resources.json")
        if (root / "test_resources.json").exists()
        else []
    )
    frames = {}
    events, artists = load_music(root)
    for day in study.TEST:
        if (root / "cache" / f"day{day}.parquet").exists():
            frames[day] = prepare_music(events, artists, day, root, False)
        else:
            with measurement(resources, f"test/prepare/day{day}"):
                frames[day] = prepare_music(events, artists, day, root, False)
        write_json(root / "test_resources.json", resources)
    evaluate(
        root,
        (*study.TRAIN, study.SEARCH, study.CONFIRM),
        frames,
        recipe,
        root / "test",
        study.SEEDS,
        resources,
        False,
    )
    if support_root is not None:
        waiting = time.monotonic()
        while not (support_root / "predictions_ready.json").exists():
            if time.monotonic() - waiting > 7200:
                raise TimeoutError("Secondary pre-test predictions did not complete")
            time.sleep(2)
        ready = study.load_json(support_root / "predictions_ready.json")
        if ready["main_test_targets_exist"] or ready["base_recipe_sha256"] != digest:
            raise ValueError("Secondary predictions do not match the pre-test freeze")
    if file_hash(root / "frozen_recipe.json") != digest:
        raise ValueError("Recipe changed during inference")
    truths = {}
    for day, frame in frames.items():
        truth = music_truth(events, day)
        truths[day] = truth[truth.user_id.isin(frame.user_id)]
        truths[day].to_parquet(root / "test" / f"day{day}_truth.parquet", index=False)
    study.summarize_output(root / "test", truths, recipe)
    write_json(root / "test_resources.json", resources)
    write_json(
        root / "test/complete.json",
        {"recipe_sha256": digest, "results_sha256": file_hash(root / "test/results.json")},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["development", "confirm", "test"])
    parser.add_argument("--root", type=Path, default=Path("runs/v3"))
    parser.add_argument("--support-root", type=Path)
    args = parser.parse_args()
    if args.phase == "test":
        test(args.root, args.support_root)
    else:
        {"development": development, "confirm": confirm}[args.phase](args.root)

"""Development-only support audit plus a controlled taste/order calibration."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from rec.io import write_json
from rec.transition import DAY, permuted_targets, transition_pairs


def expected_null(pairs):
    items = pd.Index(np.unique(np.r_[pairs.source, pairs.target]))
    groups, _ = pd.factorize(pd.MultiIndex.from_frame(pairs[["user_id", "day"]]))
    size = np.bincount(groups)
    source = items.get_indexer(pairs.source)
    target = items.get_indexer(pairs.target)
    shape = (len(size), len(items))
    left = sparse.csr_matrix((np.ones(len(pairs)), (groups, source)), shape=shape)
    right = sparse.csr_matrix((np.ones(len(pairs)), (groups, target)), shape=shape)
    expectation = (left.T @ sparse.diags(1 / size) @ right).tocsr()
    observed = sparse.csr_matrix(
        (np.ones(len(pairs)), (source, target)), shape=(len(items), len(items))
    )
    return items, observed, expectation


def synthetic(seed, rho):
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(100):
        block = user % 2
        for day in (0, 1):
            current = int(rng.integers(8))
            for _ in range(100):
                target = (current + 1) % 8 if rng.random() < rho else int(rng.integers(8))
                rows.append((user, day, current + 8 * block, target + 8 * block))
                current = target
    pairs = pd.DataFrame(rows, columns=["user_id", "day", "source", "target"])
    items, observed, background = expected_null(pairs)
    i = items.to_numpy()
    cycle = (i // 8) * 8 + (i + 1) % 8
    columns = items.get_indexer(cycle)
    cycle_excess = float(
        np.asarray((observed - background)[np.arange(len(i)), columns]).sum() / len(pairs)
    )
    outside = (
        np.asarray(
            background[
                items.get_indexer(pairs.source),
                items.get_indexer(permuted_targets(pairs, seed + 1000, False)),
            ]
        ).ravel()
        == 0
    )
    return {
        "rho": rho,
        "seed": seed,
        "cycle_excess_mass": cycle_excess,
        "global_noise_outside_context_support": float(outside.mean()),
    }


def diagnose(base, output):
    events = pd.read_parquet(base / "sampled_events.parquet").rename(columns={"uid": "user_id"})
    cutoff = 294 * DAY - 1800
    history = events[events.timestamp < cutoff]
    pairs = transition_pairs(history, cutoff)
    items, observed, expectation = expected_null(pairs)
    s, t = set(pairs.source), set(pairs.target)
    global_noise = permuted_targets(pairs, 17, False)
    mass = np.asarray(
        expectation[items.get_indexer(pairs.source), items.get_indexer(global_noise)]
    ).ravel()
    row_error = float(
        np.max(np.abs(np.asarray(observed.sum(axis=1)) - np.asarray(expectation.sum(axis=1))))
    )
    column_error = float(
        np.max(np.abs(np.asarray(observed.sum(axis=0)) - np.asarray(expectation.sum(axis=0))))
    )
    candidates = pd.read_parquet(base / "cache/day294.parquet", columns=["item_id"])
    recent = (
        history[history.played_ratio_pct >= 50]
        .sort_values(["user_id", "timestamp", "item_id"])
        .groupby("user_id")
        .tail(100)
    )
    result = {
        "scope": "search day294 only; no held-out target sets",
        "pairs": len(pairs),
        "active_items": len(items),
        "context_expected_null_nnz": expectation.nnz,
        "global_noise_outside_context_support": float((mass == 0).mean()),
        "row_margin_max_error": row_error,
        "column_margin_max_error": column_error,
        "source_only_items": len(s - t),
        "target_only_items": len(t - s),
        "candidate_untrained_target_role_share": float(candidates.item_id.isin(s - t).mean()),
        "profile_untrained_source_role_share": float(recent.item_id.isin(t - s).mean()),
        "synthetic": [synthetic(seed, rho) for rho in (0.0, 0.7) for seed in range(10)],
    }
    write_json(output, result)
    endpoint_users = pd.concat(
        [
            pairs[["user_id", endpoint]].rename(columns={endpoint: "item_id"})
            for endpoint in ("source", "target")
        ],
        ignore_index=True,
    ).drop_duplicates()
    users_per_item = endpoint_users.groupby("item_id").size()
    write_json(
        output.with_name("sparsity_diagnostics.json"),
        {
            "scope": "search historical transition pairs only",
            "active_items": len(items),
            "single_user_item_fraction": float((users_per_item == 1).mean()),
            "median_distinct_users_per_item": float(users_per_item.median()),
            "distinct_directed_edges": observed.nnz,
            "singleton_edge_fraction": float((observed.data == 1).mean()),
            "median_observations_per_edge": float(np.median(observed.data)),
            "transitions": len(pairs),
        },
    )
    print({k: v for k, v in result.items() if k != "synthetic"})
    print(
        pd.DataFrame(result["synthetic"])
        .groupby("rho")[["cycle_excess_mass", "global_noise_outside_context_support"]]
        .agg(["mean", "std"])
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/v3"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v3_support/noise_diagnostics.json")
    )
    args = parser.parse_args()
    diagnose(args.base, args.output)

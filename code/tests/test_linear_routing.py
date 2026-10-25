import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from rec.candidate_ablation import original_pool
from rec.linear import HistoricalMatrix
from rec.protocol import Config, history_before
from rec.retrieval import retrieve
from rec.routing import TemporalRouter, listwise_objective


def test_ease_matches_closed_form_and_has_zero_diagonal(toy_data):
    events, _, _ = toy_data
    cutoff = pd.Timestamp("2021-02-12")
    matrix = HistoricalMatrix.build(history_before(events, cutoff), cutoff)
    coefficients = matrix.ease(10, threads=1)
    gram = (matrix.binary.T @ matrix.binary).toarray() + 10 * np.eye(len(matrix.items))
    inverse = np.linalg.inv(gram)
    expected = -inverse / np.diag(inverse)[None, :]
    np.fill_diagonal(expected, 0)
    np.testing.assert_allclose(coefficients, expected, atol=1e-7)
    assert not np.diag(coefficients).any()


def test_retrieval_query_subset_does_not_change_other_users(toy_data):
    events, items, _ = toy_data
    cutoff = pd.Timestamp("2021-02-12")
    history = history_before(events, cutoff)
    config = Config(candidates=10, retrieval_per_source=8, neighbors=5, k=3)
    first = retrieve(history, items, cutoff, config).candidates
    chosen = np.array([0, 2, 5])
    subset = retrieve(history, items, cutoff, config, query_users=chosen).candidates
    pd.testing.assert_frame_equal(first[first.user_id.isin(chosen)].reset_index(drop=True), subset)
    extras = {int(user): np.sort(history.item_id.unique())[::2] for user in chosen}
    augmented = retrieve(
        history, items, cutoff, config, query_users=chosen, extra_items=extras
    ).candidates
    pd.testing.assert_frame_equal(subset, original_pool(augmented, budget=10))


def test_ease_recommendations_exclude_seen_and_pair_scores_ignore_targets(toy_data):
    events, _, _ = toy_data
    cutoff = pd.Timestamp("2021-02-12")
    history = history_before(events, cutoff)
    matrix = HistoricalMatrix.build(history, cutoff)
    coefficients = matrix.ease(10, threads=1)
    recommendations = matrix.recommend(coefficients, np.array([0, 1]), half_life=7, count=10)
    for user, items in recommendations.items():
        assert not set(items) & set(history[history.user_id == user].item_id)
    pairs = pd.DataFrame(
        [(u, i) for u, values in recommendations.items() for i in values],
        columns=["user_id", "item_id"],
    )
    first = matrix.pair_scores(pairs, coefficients, 7)
    second = matrix.pair_scores(pairs.assign(label=1, future_item=999), coefficients, 7)
    np.testing.assert_array_equal(first, second)


def test_router_analytical_gradient_matches_finite_differences():
    rng = np.random.default_rng(42)
    context = np.column_stack([np.ones(2), rng.normal(size=(2, 2))])
    experts = rng.normal(size=(6, 3))
    ids = np.array([0, 0, 0, 1, 1, 1])
    starts, counts = np.array([0, 3]), np.array([3, 3])
    labels = np.array([1, 0, 0, 0, 1, 1])
    offset = rng.normal(size=6)
    args = (context, experts, ids, starts, counts, labels, offset, 0.5, 0.01)
    weights = rng.normal(size=9) * 0.1
    error = check_grad(
        lambda p: listwise_objective(p, *args)[0],
        lambda p: listwise_objective(p, *args)[1],
        weights,
    )
    assert error < 1e-5


def test_router_predictions_do_not_access_labels():
    frame = pd.DataFrame(
        {
            "query_id": ["a"] * 3 + ["b"] * 3,
            "label": [1, 0, 0, 0, 1, 0],
            "user_count_all": [2] * 3 + [7] * 3,
            "user_age_min": [1] * 6,
            "user_age_mean": [4] * 6,
            "user_velocity_7": [0.3] * 6,
            "user_velocity_30": [0.8] * 6,
            "expert_a": [2, 1, 0, 0, 2, 1],
            "expert_b": [1, 2, 0, 0, 2, 1],
            "base_score": [1, 0, -1, -1, 1, 0],
        }
    )
    model = TemporalRouter(["expert_a", "expert_b"], offset_column="base_score").fit(frame)
    np.testing.assert_array_equal(model.predict(frame), model.predict(frame.drop(columns="label")))
    np.testing.assert_array_equal(
        model.predict(frame), TemporalRouter.from_state(model.state()).predict(frame)
    )

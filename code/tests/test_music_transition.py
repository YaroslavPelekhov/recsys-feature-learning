import numpy as np
import pandas as pd
import pytest

from rec.music import bounded_top, music_truth, prepare_music
from rec.retrieval import top_indices
from rec.transition import (
    DAY,
    TransitionEmbedding,
    graph_operators,
    permuted_targets,
    noise_targets,
    transition_pairs,
)


def test_noise_extension_preserves_original_controls():
    pairs = pd.DataFrame(
        {
            "user_id": [1] * 5 + [2] * 5,
            "day": [0] * 10,
            "source": np.arange(10),
            "target": np.arange(10, 20),
        }
    )
    for conditional in (False, True):
        np.testing.assert_array_equal(
            noise_targets(pairs, 17, conditional), permuted_targets(pairs, 17, conditional)
        )
    mixture = noise_targets(pairs, 17, "mixture")
    local = permuted_targets(pairs, 17, True)
    global_noise = permuted_targets(pairs, 17, False)
    assert np.all((mixture == local) | (mixture == global_noise))
    assert len(mixture) == len(pairs)


def test_conditional_null_preserves_marginals_inside_each_user_day():
    pairs = pd.DataFrame(
        {
            "user_id": [1] * 6 + [2] * 4,
            "day": [0] * 3 + [1] * 3 + [0] * 4,
            "source": np.arange(10),
            "target": [1, 2, 2, 3, 4, 5, 7, 8, 7, 9],
        }
    )
    targets = permuted_targets(pairs, 19, True)
    assert np.array_equal(targets, permuted_targets(pairs, 19, True))
    for indices in pairs.groupby(["user_id", "day"]).indices.values():
        np.testing.assert_array_equal(
            np.sort(targets[indices]), np.sort(pairs.target.to_numpy()[indices])
        )
    assert not np.array_equal(targets, pairs.target.to_numpy())


def test_transition_graph_rejects_future_and_excludes_zero_gap():
    history = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2],
            "item_id": [1, 2, 3, 4, 5],
            "timestamp": np.array([0, 0, 10, 3000, 10], dtype=np.uint32),
            "played_ratio_pct": [100] * 5,
        }
    )
    pairs = transition_pairs(history, 4000)
    assert list(zip(pairs.source, pairs.target)) == [(2, 3)]
    with pytest.raises(ValueError, match="strictly historical"):
        transition_pairs(history, 3000)


def test_signed_graph_vanishes_when_conditional_null_equals_observations():
    pairs = pd.DataFrame({"source": [1, 2], "target": [3, 4], "user_id": [1, 2], "day": [0, 0]})
    operators = graph_operators(pairs, pd.Index([1, 2, 3, 4]))
    assert operators["signed"].nnz == 0
    assert operators["raw"].nnz == 2


def test_partial_top_k_preserves_exact_item_id_ties():
    rng = np.random.default_rng(17)
    ids = rng.permutation(1000)
    for scores in (np.zeros(1000), rng.integers(-2, 3, 1000).astype(float), rng.normal(size=1000)):
        np.testing.assert_array_equal(bounded_top(scores, ids, 25), top_indices(scores, ids, 25))


def test_music_truth_retains_repeat_items_and_uses_exclusive_end():
    events = pd.DataFrame(
        {
            "user_id": [1] * 5,
            "item_id": [7, 7, 8, 9, 10],
            "timestamp": [DAY - 1, DAY, DAY + 1, 2 * DAY, DAY + 2],
            "played_ratio_pct": [100, 100, 49, 100, 50],
        }
    )
    truth = music_truth(events, 1)
    assert set(truth.item_id) == {7, 10}


def test_embedding_pair_scoring_uses_zero_for_untrained_items():
    model = TransitionEmbedding(
        pd.Index([10, 20]), np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32), [], {}
    )
    values = model.pair_scores(
        np.array([[1, 2]], np.float32), np.array([0, 0]), np.array([10, 999])
    )
    assert values[0] == pytest.approx(1 / np.sqrt(2))
    assert values[1] == 0


def test_music_pipeline_future_changes_do_not_affect_features_or_candidates(tmp_path, monkeypatch):
    def fake_fit(cls, pairs, conditional, seed):
        items = pd.Index(np.unique(np.r_[pairs.source, pairs.target]))
        matrix = np.eye(len(items), dtype=np.float32)
        return TransitionEmbedding(items, matrix, matrix, [{"epoch": 1, "loss": 0.5}], {})

    monkeypatch.setattr(TransitionEmbedding, "fit", classmethod(fake_fit))
    history = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 20, 10, 20],
            "timestamp": np.array([100, 200, 110, 210], dtype=np.uint32),
            "played_ratio_pct": [100] * 4,
            "artist": [1, 2, 1, 2],
        }
    )
    future = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [99],
            "timestamp": np.array([DAY + 1], dtype=np.uint32),
            "played_ratio_pct": [100],
            "artist": [3],
        }
    )
    artists = pd.Series({10: 1, 20: 2, 99: 3, 777: 4})
    first = prepare_music(pd.concat([history, future]), artists, 1, tmp_path / "first", False)
    changed = future.assign(item_id=777, played_ratio_pct=0, artist=4)
    second = prepare_music(pd.concat([history, changed]), artists, 1, tmp_path / "second", False)
    pd.testing.assert_frame_equal(first, second)
    assert np.isfinite(first.select_dtypes(include="number").to_numpy()).all()

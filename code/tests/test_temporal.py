import numpy as np
import pandas as pd
import pytest

from rec.data import sample_users
from rec.features import FeatureBuilder, Kernel
from rec.protocol import Config, Fold, future_truth, history_before, temporal_folds
from rec.retrieval import retrieve


def small_config(**kwargs):
    return Config(
        candidates=10,
        retrieval_per_source=8,
        neighbors=5,
        k=3,
        search_windows=(3,),
        search_half_lives=(),
        **kwargs,
    )


def test_future_perturbations_do_not_change_features_or_candidates(toy_data):
    events, items, users = toy_data
    cutoff = pd.Timestamp("2021-02-12")
    config = small_config()
    history = history_before(events, cutoff)
    first = retrieve(history, items, cutoff, config).candidates
    features, groups = FeatureBuilder(history, items, users, cutoff).build(first, config)
    changed = events.copy()
    changed.loc[changed.timestamp >= cutoff, "item_id"] = 99999
    changed = pd.concat(
        [changed, pd.DataFrame({"user_id": [9999], "item_id": [88888], "timestamp": [cutoff]})],
        ignore_index=True,
    )
    history2 = history_before(changed, cutoff)
    second = retrieve(history2, items, cutoff, config).candidates
    features2, groups2 = FeatureBuilder(history2, items, users, cutoff).build(second, config)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(features, features2)
    assert groups == groups2
    assert 99999 not in second.item_id.values
    assert 88888 not in second.item_id.values


def test_no_seen_items_and_no_future_catalog_in_candidates(toy_data):
    events, items, _ = toy_data
    cutoff = pd.Timestamp("2021-02-12")
    history = history_before(events, cutoff)
    pairs = retrieve(history, items, cutoff, small_config()).candidates
    assert pairs.merge(history, on=["user_id", "item_id"]).empty
    assert set(pairs.item_id) <= set(history.item_id)
    assert not pairs.duplicated(["user_id", "item_id"]).any()
    assert pairs.groupby("user_id").size().max() <= 10


def test_cutoff_and_label_horizon_boundaries():
    cutoff = pd.Timestamp("2021-02-01")
    end = cutoff + pd.Timedelta(days=7)
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "item_id": [10, 11, 10, 12],
            "timestamp": [
                cutoff - pd.Timedelta(days=1),
                cutoff,
                cutoff + pd.Timedelta(days=1),
                end,
            ],
        }
    )
    history = history_before(events, cutoff)
    assert history.item_id.tolist() == [10]
    truth = future_truth(events, Fold("test", cutoff, end), history)
    assert truth.item_id.tolist() == [11]


def test_future_rows_rejected_by_feature_and_retrieval_engines(toy_data):
    events, items, users = toy_data
    cutoff = events.timestamp.min()
    with pytest.raises(ValueError, match="historical"):
        FeatureBuilder(events, items, users, cutoff)
    with pytest.raises(ValueError, match="historical"):
        retrieve(events, items, cutoff, small_config())


def test_temporal_folds_have_disjoint_labels_and_embargo_is_not_needed():
    folds = temporal_folds(pd.Timestamp("2021-08-23"), Config())
    assert len(folds) == 7
    assert [fold.name for fold in folds][-3:] == ["search", "confirm", "test"]
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.end == later.cutoff


def test_sampling_is_over_users_and_order_independent():
    ids = np.array([1, 1, 1, 2, 3, 4, 5])
    selected = sample_users(ids, 3, 42)
    np.testing.assert_array_equal(selected, sample_users(ids[::-1], 3, 42))
    retained = ids[np.isin(ids, selected)]
    for user in selected:
        assert np.sum(retained == user) == np.sum(ids == user)


def test_kernel_boundaries_and_half_life():
    np.testing.assert_array_equal(Kernel("window", 7).weights(np.array([1, 7, 8])), [1, 1, 0])
    np.testing.assert_allclose(Kernel("decay", 10).weights(np.array([0, 10, 20])), [1, 0.5, 0.25])


def test_multi_genre_event_has_unit_total_mass(toy_data):
    _, items, users = toy_data
    cutoff = pd.Timestamp("2021-02-01")
    history = pd.DataFrame(
        {"user_id": [0], "item_id": [1], "timestamp": [cutoff - pd.Timedelta(days=1)]}
    )
    builder = FeatureBuilder(history, items, users, cutoff)
    block = builder.aggregate(
        pd.DataFrame({"user_id": [0], "item_id": [3]}), Kernel("window", None)
    )
    assert block.loc[0, "user_count_all"] == 1
    assert block.loc[0, "ui_genres_sum_all"] == 1
    assert block.loc[0, "ui_genres_max_all"] == 0.5

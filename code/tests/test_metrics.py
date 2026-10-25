import numpy as np
import pandas as pd
import pytest

from rec.metrics import paired_bootstrap, user_metrics
from rec.retrieval import top_indices


def test_unretrieved_positives_remain_in_denominator():
    pairs = pd.DataFrame({"user_id": [1, 1], "item_id": [10, 20]})
    truth = pd.DataFrame({"user_id": [1, 1], "item_id": [10, 999]})
    metrics = user_metrics(pairs, np.array([1.0, 0.0]), truth, np.array([1]), k=2).iloc[0]
    assert metrics.recall == 0.5
    assert metrics.candidate_recall == 0.5
    assert metrics.ndcg == pytest.approx(1 / (1 + 1 / np.log2(3)))
    assert metrics.conditional_ndcg == 1
    assert metrics.oracle_ndcg == metrics.ndcg


def test_user_with_no_retrieved_target_scores_zero():
    pairs = pd.DataFrame({"user_id": [1], "item_id": [10]})
    truth = pd.DataFrame({"user_id": [1, 2], "item_id": [999, 998]})
    metrics = user_metrics(pairs, np.array([1.0]), truth, np.array([1, 2]), k=10)
    assert len(metrics) == 2
    assert metrics.ndcg.sum() == 0
    assert metrics.recall.sum() == 0


def test_ties_are_broken_by_item_id():
    np.testing.assert_array_equal(
        top_indices(np.array([1.0, 1.0, 1.0]), np.array([30, 10, 20]), 2), [1, 2]
    )
    pairs = pd.DataFrame({"user_id": [1, 1], "item_id": [20, 10]})
    truth = pd.DataFrame({"user_id": [1], "item_id": [10]})
    assert user_metrics(pairs, np.array([1.0, 1.0]), truth, np.array([1]), 1).iloc[0].ndcg == 1


def test_duplicate_candidates_fail():
    pairs = pd.DataFrame({"user_id": [1, 1], "item_id": [10, 10]})
    with pytest.raises(ValueError, match="Duplicate"):
        user_metrics(pairs, np.array([1.0, 1.0]), pairs, np.array([1]))


def test_bootstrap_is_paired_and_deterministic():
    first = pd.DataFrame({"user_id": [1, 2, 3], "ndcg": [0.2, 0.3, 0.1]})
    second = pd.DataFrame({"user_id": [3, 1, 2], "ndcg": [0.2, 0.3, 0.4]})
    result = paired_bootstrap(first, second, 42, 200)
    assert result["difference_b_minus_a"] == pytest.approx(0.1)
    np.testing.assert_allclose(result["ci95"], [0.1, 0.1])
    assert result == paired_bootstrap(first, second, 42, 200)
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(first, second.iloc[:2], 42, 20)

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from rec.reconstruction import LOG_FLOOR, pool_experts, reconstructed_scores
from rec.transition import DAY, TransitionEmbedding


def test_exact_critic_background_restoration_recovers_mixture():
    probability = np.array([[0.8, 0.1, 0.1], [0.05, 0.15, 0.8]])
    background = np.array([0.7, 0.2, 0.1])
    weights = np.array([0.3, 0.7])
    critic = np.log(probability) - np.log(background)
    result = pool_experts(critic, weights, np.log(background))
    np.testing.assert_allclose(np.exp(result["A1"]), weights @ probability, atol=1e-14)
    expected_geometric = weights @ np.log(probability)
    np.testing.assert_allclose(result["G1"], expected_geometric, atol=1e-14)
    np.testing.assert_allclose(softmax(result["G0"]), softmax(weights @ critic), atol=1e-14)


def test_arithmetic_and_geometric_can_prefer_different_targets():
    probability = np.array([[0.80, 0.19, 0.01], [0.01, 0.19, 0.80]])
    result = pool_experts(np.log(probability), [0.5, 0.5], np.zeros(3))
    assert np.argmax(result["G0"]) == 1
    assert np.argmax(result["A0"]) in (0, 2)


def test_probability_pooling_is_invariant_to_source_normalizers_and_weight_scale():
    logits = np.array([[1.0, 2.0, 3.0], [4.0, 1.0, -1.0]])
    first = pool_experts(logits, [1, 3], np.log([0.1, 0.2, 0.7]))
    second = pool_experts(logits + np.array([[19.0], [-7.0]]), [10, 30], np.log([1.0, 2.0, 7.0]))
    for key in first:
        np.testing.assert_allclose(first[key], second[key], atol=1e-13)


def test_role_masks_and_empty_history_fallback_use_only_observed_endpoint_roles():
    history = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "item_id": [10, 20, 30, 999],
            "timestamp": np.array([100, 200, 300, 100], dtype=np.uint32),
            "played_ratio_pct": [100] * 4,
        }
    )
    model = TransitionEmbedding(
        pd.Index([10, 20, 30]), np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32), [], {}
    )
    candidates = pd.DataFrame({"user_id": [1, 1, 1, 2, 2], "item_id": [10, 20, 30, 20, 30]})
    result, audit = reconstructed_scores(candidates, history, model, DAY)
    assert audit["empty_source_queries"] == 1
    assert audit["unsupported_candidate_pairs"] == 1
    for name in ("G0", "G1", "A0", "A1"):
        assert result.loc[0, name] == LOG_FLOOR
        np.testing.assert_allclose(result.loc[[3, 4], name], np.log([0.5, 0.5]), atol=1e-14)
    with pytest.raises(ValueError, match="strictly historical"):
        reconstructed_scores(candidates, history, model, 200)

import numpy as np
import pandas as pd

from rec.experiment import fit_model, make_pool, prepare_fold, score_model
from rec.metrics import user_metrics
from rec.protocol import Config, Fold


def test_temporal_training_to_prediction_on_known_fixture(toy_data, tmp_path):
    events, items, users = toy_data
    config = Config(
        candidates=12,
        retrieval_per_source=12,
        neighbors=5,
        k=3,
        iterations=5,
        threads=2,
        depths=(3,),
        model_seeds=(17,),
        search_windows=(3,),
        search_half_lives=(),
    )
    train_fold = Fold("train", pd.Timestamp("2021-02-05"), pd.Timestamp("2021-02-12"))
    test_fold = Fold("test", pd.Timestamp("2021-02-12"), pd.Timestamp("2021-02-19"))
    train, _, groups = prepare_fold(events, items, users, train_fold, config, tmp_path)
    test, truth, _ = prepare_fold(events, items, users, test_fold, config, tmp_path)
    assert train.label.sum() > 0
    assert test.label.sum() > 0
    columns = groups["manual"]
    assert {"label", "query_id", "user_id", "item_id"}.isdisjoint(columns)
    assert np.isfinite(train[columns].select_dtypes("number").to_numpy()).all()
    model = fit_model(train, columns, config, 3, 17)
    scores = score_model(model, test, columns)
    np.testing.assert_array_equal(scores, model.predict(make_pool(test, columns)))
    # Inference must be possible with no labels or query identifiers present at all.
    np.testing.assert_array_equal(
        scores, score_model(model, test.drop(columns=["label", "query_id"]), columns)
    )
    metrics = user_metrics(test, scores, truth, truth.user_id.unique(), 3)
    assert np.isfinite(scores).all()
    assert len(metrics) == truth.user_id.nunique()
    assert metrics.ndcg.between(0, 1).all()

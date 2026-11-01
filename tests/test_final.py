import numpy as np
import pandas as pd
import pytest

from fraud.final import ensemble_predict, train_ensemble, train_fixed


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 2_000
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2 * x - 2)))).astype(np.int8)
    return pd.DataFrame({"x": x.astype(np.float32), "noise": rng.normal(size=n), "isFraud": y})


def test_train_fixed_builds_exactly_n_trees(data):
    model = train_fixed(data, ["x", "noise"], n_estimators=17)
    assert model.get_booster().num_boosted_rounds() == 17


def test_ensemble_is_the_mean_of_its_models(data):
    features = ["x", "noise"]
    models = train_ensemble(data, features, n_estimators=20, seeds=(0, 1))
    each = [m.predict_proba(data[features])[:, 1] for m in models]
    assert not np.allclose(each[0], each[1])  # seeds differ (row/column sampling)
    np.testing.assert_allclose(ensemble_predict(models, data, features), (each[0] + each[1]) / 2)

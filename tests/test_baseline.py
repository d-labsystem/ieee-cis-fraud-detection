import numpy as np
import pandas as pd
import pytest

from fraud.baseline import NON_FEATURES, feature_columns, train_baseline
from fraud.data.load import SECONDS_PER_DAY, add_time_features
from fraud.split import time_split


@pytest.fixture
def split():
    """3,000 transactions over days 1-150; fraud depends on `signal` and `product`."""
    rng = np.random.default_rng(0)
    n = 3_000
    signal = rng.normal(size=n)
    product = rng.choice(["C", "W"], size=n)
    logit = 2 * signal + np.where(product == "C", 1.5, -1.5) - 2
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.sort(rng.integers(SECONDS_PER_DAY, 151 * SECONDS_PER_DAY, n)),
            "isFraud": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8),
            "signal": signal.astype(np.float32),
            "noise": rng.normal(size=n).astype(np.float32),
            "product": pd.Categorical(product),
        }
    )
    df = add_time_features(df)
    return time_split(df, train_months=(0, 1, 2), valid_months=(3,), test_months=(4,))


def test_feature_columns_drop_id_time_and_label():
    df = pd.DataFrame(columns=[*NON_FEATURES, "TransactionAmt", "card1"])
    assert feature_columns(df) == ["TransactionAmt", "card1"]


def test_baseline_learns_signal(split):
    result = train_baseline(split, verbose=False, n_estimators=50, early_stopping_rounds=10)
    assert result.features == ["signal", "noise", "product"]
    assert 0 <= result.best_iteration < 50
    assert result.valid_scores.n_transactions == len(split.valid)
    assert result.valid_prob.shape == (len(split.valid),)
    assert result.valid_scores.roc_auc > 0.85
    assert result.valid_scores.pr_lift > 1.5

import numpy as np
import pandas as pd
import pytest

from fraud.baseline import train_baseline
from fraud.data.load import SECONDS_PER_DAY, add_time_features
from fraud.explain import BIAS, family_importance, feature_family, global_importance, shap_values
from fraud.split import time_split


@pytest.fixture(scope="module")
def trained():
    """Fraud rises with `signal`, falls with `protect`; `noise` is irrelevant."""
    rng = np.random.default_rng(0)
    n = 3_000
    signal, protect = rng.normal(size=n), rng.normal(size=n)
    signal[rng.random(n) < 0.2] = np.nan  # some missing values
    logit = 2 * np.nan_to_num(signal) - 1.5 * protect - 2.5
    df = add_time_features(
        pd.DataFrame(
            {
                "TransactionID": np.arange(n),
                "TransactionDT": np.sort(rng.integers(SECONDS_PER_DAY, 151 * SECONDS_PER_DAY, n)),
                "isFraud": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8),
                "signal": signal.astype(np.float32),
                "protect": protect.astype(np.float32),
                "noise": rng.normal(size=n).astype(np.float32),
                "ProductCD": pd.Categorical(rng.choice(["C", "W"], size=n)),
            }
        )
    )
    split = time_split(df, train_months=(0, 1, 2), valid_months=(3,), test_months=(4,))
    result = train_baseline(split, verbose=False, n_estimators=200, early_stopping_rounds=20)
    shap = shap_values(result.model, split.valid, result.features, result.best_iteration)
    return result, split, shap


def test_shap_adds_up_to_model_score(trained):
    """bias + sum of SHAP values = log-odds of predict_proba (best iteration)."""
    result, split, shap = trained
    log_odds = shap.sum(axis=1).to_numpy()
    np.testing.assert_allclose(1 / (1 + np.exp(-log_odds)), result.valid_prob, rtol=1e-4, atol=1e-6)


def test_shap_shape_and_columns(trained):
    result, split, shap = trained
    assert list(shap.columns) == [*result.features, BIAS]
    assert shap.index.equals(split.valid.index)


def test_importance_ranks_and_directions(trained):
    result, split, shap = trained
    imp = global_importance(shap, split.valid)
    assert set(imp.index[:2]) == {"signal", "protect"}
    assert imp.loc["noise", "rank"] > 2
    assert imp.loc["signal", "direction"] > 0.5
    assert imp.loc["protect", "direction"] < -0.5
    assert np.isnan(imp.loc["ProductCD", "direction"])  # categorical: no direction
    assert imp["share_%"].sum() == pytest.approx(100)
    assert imp.loc["signal", "missing"] == pytest.approx(split.valid["signal"].isna().mean())


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("uid_amt_ratio", "4a client history"),
        ("D1n", "4c normalised D (incl. D1n)"),
        ("D15n", "4c normalised D (incl. D1n)"),
        ("card1_share_30d", "4b entity frequency"),
        ("P_email_n_24h", "4b entity frequency"),
        ("hour", "4e hour"),
        ("V258", "raw V"),
        ("D3", "raw D (gaps)"),
        ("id_31", "raw identity/device"),
        ("DeviceInfo", "raw identity/device"),
        ("P_emaildomain", "raw card/address/email"),
        ("card1", "raw card/address/email"),
        ("TransactionAmt", "raw amount/product"),
    ],
)
def test_feature_family(name, family):
    assert feature_family(name) == family


def test_family_shares_sum_to_100(trained):
    result, split, shap = trained
    fam = family_importance(global_importance(shap, split.valid))
    assert fam["share_pct"].sum() == pytest.approx(100)

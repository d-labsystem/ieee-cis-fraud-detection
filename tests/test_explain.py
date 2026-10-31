import numpy as np
import pandas as pd
import pytest

from fraud.baseline import train_baseline
from fraud.data.load import SECONDS_PER_DAY, add_time_features
from fraud.explain import (
    BIAS,
    category_effects,
    family_importance,
    feature_family,
    global_importance,
    pick_cases,
    reasons,
    shap_values,
)
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
        ("browser_versions_behind", "4f browser/OS/device"),
        ("os", "4f browser/OS/device"),
        ("device_brand", "4f browser/OS/device"),
        ("device_share_30d", "4b entity frequency"),
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


# ---------- local explanations ----------

def test_category_effects_counts_and_means():
    X = pd.DataFrame({"email": pd.Categorical(["a", "a", "b", None])})
    shap = pd.DataFrame({"email": [0.2, 0.4, -1.0, 0.5], BIAS: 0.0})
    out = category_effects(shap, X, pd.Series([1, 0, 0, 1]), "email")
    assert out.loc["a"].tolist() == pytest.approx([2, 0.5, 0.3])
    assert out.loc["<missing>", "mean_shap"] == pytest.approx(0.5)
    assert out.index[0] == "a"  # most common first


def test_reasons_sorted_by_absolute_push():
    shap_row = pd.Series({"amt": 0.3, "hour": -0.9, "email": 0.1, BIAS: -3.0})
    x_row = pd.Series({"amt": 450.0, "hour": np.nan, "email": "gmail.com"})
    out = reasons(shap_row, x_row, top=2)
    assert list(out.index) == ["hour", "amt"]
    assert out.loc["hour", "value"] == "missing"
    assert out.loc["hour", "direction"] == "-> less fraud-like"
    assert out.loc["amt", "value"] == "450"


def test_pick_cases():
    pred = pd.DataFrame({
        "prob":    [0.99, 0.95, 0.90, 0.50, 0.10, 0.05, 0.04, 0.01, 0.30, 0.02],
        "isFraud": [1,    0,    1,    0,    1,    0,    1,    0,    0,    1],
    })
    cases = pick_cases(pred, alert_share=0.3)  # threshold = top 3 scores
    assert cases["caught fraud"] == 0
    assert cases["false alarm"] == 1
    # Missed frauds (below threshold): scores 0.10, 0.04, 0.02 -> median 0.04.
    assert cases["missed fraud"] == 6

import numpy as np
import pandas as pd
import pytest

from fraud.features.missing import add_missing_features
from fraud.features.vreduce import drop_correlated_v, select_v_columns, v_blocks

NAN = np.nan


# ---------- add_missing_features ----------


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "V1": [1.0, NAN, NAN],
            "V2": [1.0, NAN, 3.0],
            "D1": [NAN, 1.0, 2.0],
            "D1n": [NAN, NAN, NAN],  # engineered: must not be counted
            "M1": ["T", None, "F"],
            "id_01": [NAN, -5.0, NAN],
            "id_02": [NAN, NAN, NAN],
            "card1": [1, 2, 3],
            "uid_n_prev": [NAN, NAN, NAN],  # engineered: must not be counted
        }
    )


def test_family_counts(raw):
    out = add_missing_features(raw)
    assert out["n_missing_V"].tolist() == [0, 2, 1]
    assert out["n_missing_D"].tolist() == [1, 0, 0]
    assert out["n_missing_M"].tolist() == [0, 1, 0]
    assert out["n_missing_id"].tolist() == [2, 1, 2]


def test_raw_total_ignores_engineered_columns(raw):
    out = add_missing_features(raw)
    assert out["n_missing_raw"].tolist() == [3, 4, 3]


def test_has_identity(raw):
    assert add_missing_features(raw)["has_identity"].tolist() == [0, 1, 0]


# ---------- V reduction ----------


@pytest.fixture
def v() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(size=n)
    other = rng.normal(size=n)
    block_b = np.where(np.arange(n) % 2 == 0, NAN, 1.0)  # missing on even rows
    return pd.DataFrame(
        {
            "V1": base,
            "V2": 2 * base + 1,  # exact copy of V1 (corr 1)
            "V3": other,  # same block, independent
            "V4": np.full(n, 7.0),  # constant
            "V5": block_b * rng.normal(size=n),  # other block
            "V6": block_b * base,  # other block: base on odd rows only
            "TransactionAmt": base,  # not a V column: never touched
            "month": np.repeat([0, 4], n // 2),
        }
    )


def test_blocks_follow_missing_mask(v):
    assert v_blocks(v) == [["V1", "V2", "V3", "V4"], ["V5", "V6"]]


def test_select_drops_copies_and_constants(v):
    assert select_v_columns(v, threshold=0.9) == ["V1", "V3", "V5", "V6"]


def test_correlation_measured_on_fit_months_only(v):
    # V3 copies V1 in month 0 only; month 4 is noise. Fit on month 0 -> dropped.
    v.loc[v["month"] == 0, "V3"] = v.loc[v["month"] == 0, "V1"] * 3
    out = drop_correlated_v(v, threshold=0.9, fit_months=(0,))
    assert "V3" not in out
    assert {"V1", "TransactionAmt", "month"} <= set(out.columns)


def test_no_fit_rows_raises(v):
    with pytest.raises(ValueError, match="No rows"):
        drop_correlated_v(v, fit_months=(9,))

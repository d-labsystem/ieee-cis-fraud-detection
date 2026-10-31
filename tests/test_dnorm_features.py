import numpy as np
import pandas as pd

from fraud.features.client import add_client_features
from fraud.features.dnorm import ANCHORS, add_dnorm_features

D_COLS = [f"D{i}" for i in range(1, 16)]


def _df() -> pd.DataFrame:
    """One client on days 10, 11 and 15: anchors grow with the day, D3 is a gap."""
    days = np.array([10, 11, 15])
    df = pd.DataFrame({c: (days - 5).astype(np.float32) for c in D_COLS})
    df["D3"] = np.array([np.nan, 1, 4], dtype=np.float32)
    df["D4"] = np.array([np.nan, 6, 10], dtype=np.float32)
    df["day"] = days.astype(np.int16)
    return df


def test_anchor_becomes_constant():
    out = add_dnorm_features(_df())
    assert out["D10n"].tolist() == [5, 5, 5]


def test_missing_stays_missing():
    out = add_dnorm_features(_df())
    assert np.isnan(out["D4n"].iloc[0])
    assert out["D4n"].iloc[1:].tolist() == [5, 5]


def test_gap_columns_not_normalised():
    out = add_dnorm_features(_df())
    assert "D3n" not in out
    pd.testing.assert_series_equal(out["D3"], _df()["D3"])


def test_add_keeps_raw_columns():
    out = add_dnorm_features(_df())
    assert set(D_COLS) <= set(out.columns)


def test_replace_drops_only_anchors():
    out = add_dnorm_features(_df(), replace=True)
    assert not set(ANCHORS) & set(out.columns)
    assert {"D3", "D5", "D7", "D9", "D13", "D14"} <= set(out.columns)
    assert {f"{c}n" for c in ANCHORS} <= set(out.columns)


def test_d1n_matches_client_step():
    df = _df().assign(
        TransactionDT=lambda d: d["day"].astype(np.int64) * 86_400,
        card1=1, addr1=100.0, TransactionAmt=10.0, P_emaildomain="a.com",
    )
    from_client = add_client_features(df)["D1n"]
    pd.testing.assert_series_equal(add_dnorm_features(df)["D1n"], from_client)

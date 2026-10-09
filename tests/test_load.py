import numpy as np
import pandas as pd
import pytest

from fraud.data.load import (
    IDENTITY_PARQUET,
    TRANSACTION_PARQUET,
    add_time_features,
    downcast,
    load_raw,
)
from fraud.data.sample import IDENTITY_FILE, TRANSACTION_FILE


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionDT": [86_400, 90_000, 172_800],
            "TransactionAmt": [10.5, 1234.56, 31937.39],
        }
    )


def _identity() -> pd.DataFrame:
    return pd.DataFrame({"TransactionID": [2], "DeviceType": ["mobile"]})


# ---------- load_raw ----------


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_merge_keeps_every_transaction(tmp_path, fmt):
    if fmt == "parquet":
        _transactions().to_parquet(tmp_path / TRANSACTION_PARQUET, index=False)
        _identity().to_parquet(tmp_path / IDENTITY_PARQUET, index=False)
    else:
        _transactions().to_csv(tmp_path / TRANSACTION_FILE, index=False)
        _identity().to_csv(tmp_path / IDENTITY_FILE, index=False)

    merged = load_raw(tmp_path)

    assert len(merged) == 3
    assert merged["TransactionID"].tolist() == [1, 2, 3]
    assert merged.loc[merged["TransactionID"] == 2, "DeviceType"].item() == "mobile"
    assert merged.loc[merged["TransactionID"] != 2, "DeviceType"].isna().all()


def test_duplicate_identity_key_raises(tmp_path):
    _transactions().to_parquet(tmp_path / TRANSACTION_PARQUET, index=False)
    duplicated = pd.DataFrame({"TransactionID": [2, 2], "DeviceType": ["mobile", "desktop"]})
    duplicated.to_parquet(tmp_path / IDENTITY_PARQUET, index=False)
    with pytest.raises(pd.errors.MergeError):
        load_raw(tmp_path)


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path)


# ---------- downcast ----------


def test_downcast_dtypes():
    df = pd.DataFrame(
        {
            "small_int": [0, 1, 1],
            "id": [2_987_000, 2_987_001, 3_577_539],
            "amount": [1.5, np.nan, 3.25],
            "label": ["a", None, "b"],
        }
    )
    out = downcast(df)
    assert out["small_int"].dtype == np.int8
    assert out["id"].dtype == np.int32
    assert out["amount"].dtype == np.float32
    assert isinstance(out["label"].dtype, pd.CategoricalDtype)


def test_downcast_preserves_values():
    df = pd.DataFrame(
        {
            "amount": [0.01, 5.99, 49.95, 1234.56, 31937.39, np.nan],
            "label": ["x", "y", None, "x", "y", "x"],
        }
    )
    out = downcast(df)
    # Every amount must still round to the same cents.
    np.testing.assert_array_equal(
        out["amount"].astype(np.float64).round(2).to_numpy(),
        df["amount"].round(2).to_numpy(),
    )
    assert out["label"].astype(object).where(out["label"].notna(), None).tolist() == [
        "x", "y", None, "x", "y", "x"
    ]


def test_downcast_does_not_modify_input():
    df = pd.DataFrame({"amount": [1.5, 2.5], "n": [1, 2]})
    downcast(df)
    assert df["amount"].dtype == np.float64
    assert df["n"].dtype == np.int64


# ---------- add_time_features ----------


@pytest.mark.parametrize(
    ("seconds", "day", "month"),
    [
        (86_400, 1, 0),                  # very first second of the data
        (2 * 86_400 - 1, 1, 0),          # last second of day 1
        (2 * 86_400, 2, 0),              # first second of day 2
        (31 * 86_400 - 1, 30, 0),        # last second of day 30 -> still month 0
        (31 * 86_400, 31, 1),            # day 31 -> month 1
    ],
)
def test_time_boundaries(seconds, day, month):
    out = add_time_features(pd.DataFrame({"TransactionDT": [seconds]}))
    assert out["day"].item() == day
    assert out["month"].item() == month


def test_time_before_first_day_raises():
    with pytest.raises(ValueError, match="86,400"):
        add_time_features(pd.DataFrame({"TransactionDT": [86_399]}))
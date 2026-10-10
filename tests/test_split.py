import numpy as np
import pandas as pd
import pytest

from fraud.data.load import SECONDS_PER_DAY, add_time_features
from fraud.split import summarize, time_split

LAST_DAY = 182  # last day in the real train data


@pytest.fixture
def df() -> pd.DataFrame:
    """One transaction per day, days 1-182; every 10th day is a fraud."""
    days = np.arange(1, LAST_DAY + 1)
    return add_time_features(
        pd.DataFrame(
            {
                "TransactionID": days,
                "TransactionDT": days * SECONDS_PER_DAY,
                "isFraud": (days % 10 == 0).astype(np.int8),
            }
        )
    )


def test_default_day_ranges(df):
    split = time_split(df)
    assert (split.train["day"].min(), split.train["day"].max()) == (1, 120)
    assert (split.valid["day"].min(), split.valid["day"].max()) == (121, 150)
    # Days 181-182 (month 6) are folded into the test month.
    assert (split.test["day"].min(), split.test["day"].max()) == (151, LAST_DAY)


def test_every_row_in_exactly_one_part(df):
    split = time_split(df)
    ids = pd.concat([split.train, split.valid, split.test])["TransactionID"]
    assert ids.is_unique
    assert sorted(ids) == sorted(df["TransactionID"])


def test_unassigned_month_raises(df):
    with pytest.raises(ValueError, match=r"Months \[3\] are not assigned"):
        time_split(df, train_months=(0, 1, 2))


@pytest.mark.parametrize(
    ("train", "valid", "test"),
    [
        ((0, 1, 2, 3), (3, 4), (5, 6)),   # overlap
        ((0, 1, 2, 4), (3,), (5, 6)),     # valid before the end of train
        ((0, 1, 2, 3), (5,), (4, 6)),     # test starts before valid
    ],
)
def test_overlapping_or_out_of_order_months_raise(df, train, valid, test):
    with pytest.raises(ValueError, match="must all come after"):
        time_split(df, train, valid, test)


def test_no_months_raises(df):
    with pytest.raises(ValueError, match="No months given for valid"):
        time_split(df, valid_months=())


def test_empty_part_raises(df):
    # Like the local 50K-row sample, which only reaches day 13.
    early = df[df["day"] <= 13]
    with pytest.raises(ValueError, match="valid is empty"):
        time_split(early, train_months=(0,), valid_months=(1,), test_months=(2,))


def test_summary_numbers(df):
    train, valid, test = summarize(time_split(df))
    assert (train.name, train.n_transactions, train.n_frauds) == ("train", 120, 12)
    assert train.fraud_rate == pytest.approx(0.1)
    assert (valid.n_transactions, valid.n_frauds) == (30, 3)
    assert (test.n_transactions, test.n_frauds) == (32, 3)
    assert (test.first_day, test.last_day) == (151, LAST_DAY)

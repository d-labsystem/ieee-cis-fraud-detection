import pandas as pd
import pytest

from fraud.data.load import SECONDS_PER_DAY
from fraud.features.clock import add_hour


@pytest.mark.parametrize(
    ("seconds", "hour"),
    [
        (SECONDS_PER_DAY, 0),                 # first second of day 1
        (SECONDS_PER_DAY + 3_599, 0),         # last second of hour 0
        (SECONDS_PER_DAY + 3_600, 1),         # first second of hour 1
        (2 * SECONDS_PER_DAY - 1, 23),        # last second of day 1
        (2 * SECONDS_PER_DAY, 0),             # wraps to the next day
    ],
)
def test_hour_boundaries(seconds, hour):
    assert add_hour(pd.DataFrame({"TransactionDT": [seconds]}))["hour"].item() == hour


def test_does_not_modify_input():
    df = pd.DataFrame({"TransactionDT": [SECONDS_PER_DAY]})
    add_hour(df)
    assert list(df.columns) == ["TransactionDT"]

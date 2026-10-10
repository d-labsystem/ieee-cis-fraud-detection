"""Time-based train / validation / test split.

The Kaggle test set comes after the training period, so evaluation must also
train on the past and score the future. A random split would let the model see
later transactions from the same clients and overstate how well it works.

Months are the 30-day buckets from ``fraud.data.load.add_time_features``.
"""

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict

TRAIN_MONTHS: tuple[int, ...] = (0, 1, 2, 3)
VALID_MONTHS: tuple[int, ...] = (4,)
# Month 6 is only days 181-182 (~5K rows, ~240 frauds): too few to score on
# its own, so it is folded into the test month.
TEST_MONTHS: tuple[int, ...] = (5, 6)


@dataclass(frozen=True)
class Split:
    """The three row-disjoint parts of the train table, in time order."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


class PartSummary(BaseModel):
    """Size and prevalence of one part, so the caller can print or assert on it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    n_transactions: int
    n_frauds: int
    fraud_rate: float
    first_day: int
    last_day: int


def _check_months(parts: dict[str, tuple[int, ...]]) -> None:
    """Each part must be non-empty and lie strictly after the previous one."""
    for name, months in parts.items():
        if not months:
            raise ValueError(f"No months given for {name}.")
    names = list(parts)
    for earlier, later in zip(names, names[1:]):
        if max(parts[earlier]) >= min(parts[later]):
            raise ValueError(
                f"{later} months {parts[later]} must all come after "
                f"{earlier} months {parts[earlier]}."
            )


def time_split(
    df: pd.DataFrame,
    train_months: tuple[int, ...] = TRAIN_MONTHS,
    valid_months: tuple[int, ...] = VALID_MONTHS,
    test_months: tuple[int, ...] = TEST_MONTHS,
) -> Split:
    """Split ``df`` by its ``month`` column into train, validation and test.

    Raises ``ValueError`` if the month lists overlap or are out of order, if
    ``df`` holds a month no part claims (rows would silently disappear), or if
    any part comes out empty (e.g. when run on the small local sample).
    """
    parts = {"train": train_months, "valid": valid_months, "test": test_months}
    _check_months(parts)

    assigned = {m for months in parts.values() for m in months}
    unassigned = {int(m) for m in df["month"].unique()} - assigned
    if unassigned:
        raise ValueError(f"Months {sorted(unassigned)} are not assigned to any part.")

    split = Split(**{name: df[df["month"].isin(months)] for name, months in parts.items()})
    for name in parts:
        if getattr(split, name).empty:
            raise ValueError(f"{name} is empty: df has no rows in months {parts[name]}.")
    return split


def summarize(split: Split) -> list[PartSummary]:
    return [
        PartSummary(
            name=name,
            n_transactions=len(part),
            n_frauds=int(part["isFraud"].sum()),
            fraud_rate=float(part["isFraud"].mean()),
            first_day=int(part["day"].min()),
            last_day=int(part["day"].max()),
        )
        for name, part in (("train", split.train), ("valid", split.valid), ("test", split.test))
    ]

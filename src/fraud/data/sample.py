""" time-conts dev sample of IEEE-CIS train data."""

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"
SECONDS_PER_DAY = 86_400


class SampleSummary(BaseModel):
    """What make_sample() produced, so the caller can print or assert on it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_transactions: int
    n_identity: int
    identity_coverage: float
    fraud_rate: float
    first_day: float
    last_day: float


def make_sample(raw_dir: Path, out_dir: Path, n_transactions: int) -> SampleSummary:
    if n_transactions <= 0:
        raise ValueError(f"n_transactions must be positive, got {n_transactions}")

    transactions = pd.read_csv(raw_dir / TRANSACTION_FILE, nrows=n_transactions)

    if not transactions["TransactionDT"].is_monotonic_increasing:
        raise ValueError(
            f"{TRANSACTION_FILE} is not sorted by TransactionDT; "
            "the first N rows are not the earliest N transactions."
        )

    # Identity rows exist for only a subset of transactions, so its row order
    # does not line up with the transaction file. Select by key, not position.
    identity = pd.read_csv(raw_dir / IDENTITY_FILE)
    identity = identity[identity["TransactionID"].isin(transactions["TransactionID"])]

    if not identity["TransactionID"].is_unique:
        raise ValueError(f"{IDENTITY_FILE} has duplicate TransactionIDs.")

    out_dir.mkdir(parents=True, exist_ok=True)
    transactions.to_parquet(out_dir / "transactions.parquet", index=False)
    identity.to_parquet(out_dir / "identity.parquet", index=False)

    return SampleSummary(
        n_transactions=len(transactions),
        n_identity=len(identity),
        identity_coverage=len(identity) / len(transactions),
        fraud_rate=float(transactions["isFraud"].mean()),
        first_day=float(transactions["TransactionDT"].min() / SECONDS_PER_DAY),
        last_day=float(transactions["TransactionDT"].max() / SECONDS_PER_DAY),
    )
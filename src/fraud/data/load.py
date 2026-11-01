import numpy as np
import pandas as pd

from pathlib import Path
from fraud.data.sample import IDENTITY_FILE, TRANSACTION_FILE

TRANSACTION_PARQUET = "transactions.parquet"
IDENTITY_PARQUET = "identity.parquet"
KEY = "TransactionID"
TEST_TRANSACTION_FILE = "test_transaction.csv"
TEST_IDENTITY_FILE = "test_identity.csv"

SECONDS_PER_DAY = 86_400
DAYS_PER_MONTH = 30
FIRST_DAY = 1


def load_raw(raw_dir: Path) -> pd.DataFrame:
    if (raw_dir / TRANSACTION_PARQUET).exists() and (raw_dir / IDENTITY_PARQUET).exists():
        transactions = pd.read_parquet(raw_dir / TRANSACTION_PARQUET)
        identity = pd.read_parquet(raw_dir / IDENTITY_PARQUET)
    elif (raw_dir / TRANSACTION_FILE).exists() and (raw_dir / IDENTITY_FILE).exists():
        transactions = pd.read_csv(raw_dir / TRANSACTION_FILE)
        identity = pd.read_csv(raw_dir / IDENTITY_FILE)
    else:
        raise FileNotFoundError(
            f"No transaction/identity parquet or CSV pair found in {raw_dir}"
        )

    # raises MergeError if either side has a duplicate key, which
    return transactions.merge(identity, on=KEY, how="left", validate="one_to_one")


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with smaller dtypes; values are preserved.

    - float64 -> float32 (about 7 significant digits; cents stay exact up to
      about 65,000, comfortably above the largest TransactionAmt)
    - int64   -> the smallest integer type that holds the column's range
    - strings -> category (each distinct value stored once)
    """
    out = df.copy(deep=False)
    for name in out.columns:
        col = out[name]
        if isinstance(col.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_float_dtype(col):
            out[name] = col.astype(np.float32)
        elif pd.api.types.is_integer_dtype(col):
            out[name] = pd.to_numeric(col, downcast="integer")
        elif pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
            out[name] = col.astype("category")
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``day`` (1-based) and ``month`` (0-based, 30-day buckets).

    30-day buckets rather than calendar months: the true start date is not
    disclosed, and inferring one is an assumption we do not need to make.
    """
    dt = df["TransactionDT"]
    if (dt < FIRST_DAY * SECONDS_PER_DAY).any():
        raise ValueError("TransactionDT below 86,400: earlier than the data's first day.")

    out = df.copy(deep=False)
    out["day"] = (dt // SECONDS_PER_DAY).astype(np.int16)
    out["month"] = ((out["day"] - FIRST_DAY) // DAYS_PER_MONTH).astype(np.int8)
    return out


def load_dataset(raw_dir: Path) -> pd.DataFrame:
    return add_time_features(downcast(load_raw(raw_dir)))


def load_competition(raw_dir: Path) -> pd.DataFrame:
    """Kaggle's train and test files as one table, test rows with isFraud NaN.

    One table because the feature steps look back in time: a test row's client
    and entity features need that client's training rows. Downcasting after the
    concat gives train and test the same category codes.
    """
    train = load_raw(raw_dir)
    test_transactions = pd.read_csv(raw_dir / TEST_TRANSACTION_FILE)
    # The test identity file names its columns id-01 ... id-38, train id_01 ...
    test_identity = pd.read_csv(raw_dir / TEST_IDENTITY_FILE).rename(
        columns=lambda c: c.replace("id-", "id_"))
    test = test_transactions.merge(test_identity, on=KEY, how="left", validate="one_to_one")

    if set(test.columns) != set(train.columns) - {"isFraud"}:
        raise ValueError("Test columns differ from train columns (other than isFraud).")
    if test["TransactionDT"].min() <= train["TransactionDT"].max():
        raise ValueError("Test transactions must all come after the training ones.")
    both = pd.concat([train, test[train.columns.drop("isFraud")]], ignore_index=True)
    return add_time_features(downcast(both))
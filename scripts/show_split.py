"""Print size and fraud rate of the train / validation / test split.

Usage (on Kaggle, after build_dataset.py --source raw):
    python scripts/show_split.py
    python scripts/show_split.py --path /kaggle/input/<your-notebook>/train_raw.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from fraud.config import get_settings
from fraud.split import summarize, time_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Parquet file to split (default: <processed_dir>/train_raw.parquet).",
    )
    args = parser.parse_args()
    path = args.path or get_settings().processed_dir / "train_raw.parquet"

    # Only the columns the split and summary need; parquet skips the rest.
    df = pd.read_parquet(path, columns=["TransactionID", "day", "month", "isFraud"])
    summary = pd.DataFrame([s.model_dump() for s in summarize(time_split(df))])
    print(f"Split of {path}\n")
    print(summary.set_index("name").to_string(formatters={"fraud_rate": "{:.2%}".format}))


if __name__ == "__main__":
    main()

"""Build the merged, downcast train table and print a per-month summary.

Usage:
    uv run python scripts/build_dataset.py --source sample   # Mac, 50K rows
    python scripts/build_dataset.py --source raw             # Kaggle, full data
"""

import argparse

from fraud.config import get_settings
from fraud.data.load import add_time_features, downcast, load_raw

BYTES_PER_GB = 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["sample", "raw"], required=True)
    args = parser.parse_args()

    settings = get_settings()
    in_dir = settings.sample_dir if args.source == "sample" else settings.raw_dir

    merged = load_raw(in_dir)
    before_gb = merged.memory_usage(deep=True).sum() / BYTES_PER_GB
    df = add_time_features(downcast(merged))
    del merged
    after_gb = df.memory_usage(deep=True).sum() / BYTES_PER_GB

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.processed_dir / f"train_{args.source}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Wrote:      {out_path}")
    print(f"Shape:      {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory:     {before_gb:.2f} GB -> {after_gb:.2f} GB")
    print(f"Fraud rate: {df['isFraud'].mean():.2%}")
    print("\nPer month:")
    per_month = df.groupby("month").agg(
        first_day=("day", "min"),
        last_day=("day", "max"),
        transactions=("isFraud", "size"),
        fraud_rate=("isFraud", "mean"),
    )
    print(per_month.to_string(formatters={"fraud_rate": "{:.2%}".format}))


if __name__ == "__main__":
    main()
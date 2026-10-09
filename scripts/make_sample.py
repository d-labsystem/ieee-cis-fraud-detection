"""Command-line entry point: build the development sample in data/sample/.

Usage:
    uv run python scripts/make_sample.py
    uv run python scripts/make_sample.py --n-transactions 20000
"""

import argparse

from fraud.config import get_settings
from fraud.data.sample import make_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-transactions",
        type=int,
        default=50_000,
        help="Number of earliest transactions to keep (default: 50000).",
    )
    args = parser.parse_args()

    settings = get_settings()
    summary = make_sample(
        raw_dir=settings.raw_dir,
        out_dir=settings.sample_dir,
        n_transactions=args.n_transactions,
    )

    print(f"Wrote sample to:   {settings.sample_dir}")
    print(f"Transactions:      {summary.n_transactions:,}")
    print(f"Identity rows:     {summary.n_identity:,}")
    print(f"Identity coverage: {summary.identity_coverage:.1%}")
    print(f"Fraud rate:        {summary.fraud_rate:.2%}")
    print(f"Day range:         {summary.first_day:.1f} -> {summary.last_day:.1f}")


if __name__ == "__main__":
    main()
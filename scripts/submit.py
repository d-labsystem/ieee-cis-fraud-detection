"""Kaggle submission: the frozen final recipe trained on every labelled month.

Nothing is chosen here. Features, parameters, tree count and seeds are the
ones final.py used; the only change is more training data (months 0-6, so the
test months' labels are used for training, never for a decision).

Kaggle's test period starts about 30 days after the last labelled day, so this
model faces the label delay Phase 7 measured (gap scenario, -15% PR-AUC). The
leaderboard reports ROC-AUC on that later period: a sanity check, not a better
estimate than final.py's test score.

Usage (on Kaggle, GPU, ~15 min, ~20 GB RAM):
    python scripts/submit.py --device cuda
Then submit <processed_dir>/submission/submission.csv to the competition.
"""

import argparse
from pathlib import Path

import pandas as pd

from fraud.baseline import TARGET, feature_columns
from fraud.config import get_settings
from fraud.data.load import load_competition
from fraud.features.pipeline import FINAL_STEPS, build_features, steps_name
from fraud.final import N_TREES, SEEDS, ensemble_predict, train_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Folder with the four competition CSVs (default: settings.raw_dir).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--trees", type=int, default=N_TREES)
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = args.raw_dir or settings.raw_dir
    df = build_features(load_competition(raw_dir), FINAL_STEPS)
    labelled = df[TARGET].notna()
    train, test = df[labelled], df[~labelled]
    features = feature_columns(train)
    print(f"Features: {steps_name(FINAL_STEPS)} ({len(features)}) | {args.trees} trees x seeds {SEEDS}")
    print(f"Train: {len(train):,} rows, days {train['day'].min()}-{train['day'].max()} | "
          f"Kaggle test: {len(test):,} rows, days {test['day'].min()}-{test['day'].max()}")

    models = train_ensemble(train, features, args.trees, args.device, SEEDS)
    prob = ensemble_predict(models, test, features)

    submission = pd.DataFrame({"TransactionID": test["TransactionID"].to_numpy(), TARGET: prob})
    expected = pd.read_csv(raw_dir / "sample_submission.csv", usecols=["TransactionID"])
    if set(submission["TransactionID"]) != set(expected["TransactionID"]):
        raise ValueError("Submission TransactionIDs differ from sample_submission.csv.")

    out_dir = settings.processed_dir / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_dir / "submission.csv", index=False)

    # Unlabelled sanity check: the mean score per month should stay near the
    # training fraud rate (Phase 7: mean score is flat even while skill decays).
    by_month = pd.Series(prob, index=test.index).groupby(test["month"]).agg(["size", "mean"])
    print(f"\nTraining fraud rate {train[TARGET].mean():.4f}; mean predicted probability by month:")
    print(by_month.to_string(float_format="{:.4f}".format))
    print(f"\nWrote {out_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()

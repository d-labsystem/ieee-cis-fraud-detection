"""Add feature steps, train XGBoost, score validation, log the result.

Each run appends one row to <processed_dir>/experiments.csv and prints the
whole table, so feature steps can be compared against the baseline.

Usage (on Kaggle, GPU):
    python scripts/run_experiment.py --device cuda                  # baseline
    python scripts/run_experiment.py --device cuda --steps client   # + Phase 4a
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from fraud.baseline import train_baseline
from fraud.config import get_settings
from fraud.features.client import add_client_features
from fraud.split import time_split

STEPS = {"client": add_client_features}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Parquet to train on (default: <processed_dir>/train_raw.parquet).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", nargs="*", choices=list(STEPS), default=[])
    args = parser.parse_args()

    settings = get_settings()
    path = args.path or settings.processed_dir / "train_raw.parquet"
    df = pd.read_parquet(path)
    raw_columns = set(df.columns)

    # Features are built on the full table before splitting. That is safe only
    # because every step is causal (uses earlier rows only).
    for step in args.steps:
        start = time.perf_counter()
        df = STEPS[step](df)
        print(f"Step {step!r}: {time.perf_counter() - start:.1f}s")
    new_columns = [c for c in df.columns if c not in raw_columns]

    result = train_baseline(time_split(df), device=args.device, verbose=200)
    s = result.valid_scores
    print(f"\nFeatures: {len(result.features)} ({len(new_columns)} new)")
    print(f"Valid ROC-AUC {s.roc_auc:.4f} | PR-AUC {s.pr_auc:.4f} | lift {s.pr_lift:.1f}x"
          f" | best iteration {result.best_iteration}")

    if new_columns:
        gain = pd.Series(result.model.get_booster().get_score(importance_type="gain"))
        rank = gain.rank(ascending=False).astype(int)
        new = pd.DataFrame({"gain": gain.reindex(new_columns), "rank": rank.reindex(new_columns)})
        print(f"\nNew features (rank out of {len(result.features)}; NaN = never used):")
        print(new.sort_values("gain", ascending=False).to_string(float_format="{:.1f}".format))

    log_path = settings.processed_dir / "experiments.csv"
    row = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "steps": "+".join(args.steps) or "baseline",
        "n_features": len(result.features),
        "best_iteration": result.best_iteration,
        "roc_auc": round(s.roc_auc, 4),
        "pr_auc": round(s.pr_auc, 4),
        "pr_lift": round(s.pr_lift, 2),
    }])
    log = pd.concat([pd.read_csv(log_path), row]) if log_path.exists() else row
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(log_path, index=False)
    print(f"\nAll runs ({log_path}):")
    print(log.to_string(index=False))


if __name__ == "__main__":
    main()

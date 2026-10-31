"""Train the XGBoost baseline and print its validation scores.

Usage (on Kaggle; GPU recommended, Accelerator → GPU):
    python scripts/train_baseline.py --device cuda
    python scripts/train_baseline.py --path /kaggle/input/<notebook>/train_raw.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from fraud.baseline import train_baseline
from fraud.config import get_settings
from fraud.split import summarize, time_split

TOP_FEATURES = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Parquet to train on (default: <processed_dir>/train_raw.parquet).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    settings = get_settings()
    path = args.path or settings.processed_dir / "train_raw.parquet"
    split = time_split(pd.read_parquet(path))
    for part in summarize(split):
        print(f"{part.name:<6} {part.n_transactions:>8,} rows  {part.fraud_rate:.2%} fraud")

    result = train_baseline(split, device=args.device)

    s = result.valid_scores
    print(f"\nFeatures:       {len(result.features)}")
    print(f"Best iteration: {result.best_iteration}")
    print(f"Valid ROC-AUC:  {s.roc_auc:.4f}")
    print(f"Valid PR-AUC:   {s.pr_auc:.4f}  (prevalence {s.prevalence:.2%}, lift {s.pr_lift:.1f}x)")

    # total_gain, not "gain": XGBoost's "gain" is the average per split, which
    # ranks a feature used in a few big splits above one used in thousands.
    gain = pd.Series(result.model.get_booster().get_score(importance_type="total_gain"))
    print(f"\nTop {TOP_FEATURES} features by total gain:")
    print(gain.sort_values(ascending=False).head(TOP_FEATURES).to_string(float_format="{:.1f}".format))

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.processed_dir / "baseline.json"
    result.model.save_model(out_path)
    print(f"\nSaved model:    {out_path}")


if __name__ == "__main__":
    main()

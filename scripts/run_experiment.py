"""Add feature steps, train XGBoost, score validation, log the result.

Each run appends one row to <processed_dir>/experiments.csv and prints the
whole table, so feature steps can be compared against the baseline. It also
saves validation predictions to <processed_dir>/predictions/.

Usage (on Kaggle, GPU):
    python scripts/run_experiment.py --device cuda                  # baseline
    python scripts/run_experiment.py --device cuda --steps client   # + Phase 4a
    python scripts/run_experiment.py --device cuda --seed 1         # noise check
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.baseline import TARGET, train_baseline
from fraud.config import get_settings
from fraud.features.client import add_client_features
from fraud.metrics import score
from fraud.split import time_split

STEPS = {"client": add_client_features}


def segment_scores(valid: pd.DataFrame, prob: np.ndarray, n_prev: pd.Series) -> pd.DataFrame:
    """Validation scores for new clients (no earlier transaction) vs returning ones."""
    is_new = (n_prev.loc[valid.index] == 0).to_numpy()
    y = valid[TARGET].to_numpy()
    rows = {}
    for name, mask in (("new client", is_new), ("returning client", ~is_new)):
        if 0 < y[mask].sum() < mask.sum():
            rows[name] = score(y[mask], prob[mask]).model_dump()
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Parquet to train on (default: <processed_dir>/train_raw.parquet).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", nargs="*", choices=list(STEPS), default=[])
    parser.add_argument("--seed", type=int, default=0,
                        help="XGBoost random_state; rerun with other seeds to measure noise.")
    args = parser.parse_args()

    settings = get_settings()
    path = args.path or settings.processed_dir / "train_raw.parquet"
    df = pd.read_parquet(path)
    raw_columns = set(df.columns)
    # Client history for the segment report only (not a feature unless the
    # "client" step adds it), so every run, the baseline too, gets the report.
    n_prev = add_client_features(df)["uid_n_prev"]

    # Features are built on the full table before splitting. That is safe only
    # because every step is causal (uses earlier rows only).
    for step in args.steps:
        start = time.perf_counter()
        df = STEPS[step](df)
        print(f"Step {step!r}: {time.perf_counter() - start:.1f}s")
    new_columns = [c for c in df.columns if c not in raw_columns]

    split = time_split(df)
    result = train_baseline(split, device=args.device, verbose=200, random_state=args.seed)
    s = result.valid_scores
    print(f"\nFeatures: {len(result.features)} ({len(new_columns)} new)")
    print(f"Valid ROC-AUC {s.roc_auc:.4f} | PR-AUC {s.pr_auc:.4f} | lift {s.pr_lift:.1f}x"
          f" | best iteration {result.best_iteration}")

    print("\nBy client history:")
    segments = segment_scores(split.valid, result.valid_prob, n_prev)
    print(segments[["n_transactions", "n_frauds", "prevalence", "roc_auc", "pr_auc", "pr_lift"]]
          .to_string(float_format="{:.4f}".format))

    if new_columns:
        # total_gain, not "gain": XGBoost's "gain" is the average per split.
        gain = pd.Series(result.model.get_booster().get_score(importance_type="total_gain"))
        share = 100 * gain / gain.sum()
        rank = gain.rank(ascending=False).astype(int)
        new = pd.DataFrame({"gain_%": share.reindex(new_columns), "rank": rank.reindex(new_columns)})
        print(f"\nNew features (share of total gain; rank out of {len(result.features)}; "
              f"NaN = never used). All new together: {new['gain_%'].sum():.1f}%")
        print(new.sort_values("gain_%", ascending=False).to_string(float_format="{:.2f}".format))

    steps = "+".join(args.steps) or "baseline"
    pred_dir = settings.processed_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "TransactionID": split.valid["TransactionID"].to_numpy(),
        TARGET: split.valid[TARGET].to_numpy(),
        "prob": result.valid_prob,
    }).to_parquet(pred_dir / f"{steps}_seed{args.seed}.parquet", index=False)

    log_path = settings.processed_dir / "experiments.csv"
    row = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "steps": steps,
        "seed": args.seed,
        "n_features": len(result.features),
        "best_iteration": result.best_iteration,
        "roc_auc": round(s.roc_auc, 4),
        "pr_auc": round(s.pr_auc, 4),
        "pr_lift": round(s.pr_lift, 2),
    }])
    log = pd.concat([pd.read_csv(log_path), row]) if log_path.exists() else row
    log.to_csv(log_path, index=False)
    print(f"\nAll runs ({log_path}):")
    print(log.to_string(index=False))


if __name__ == "__main__":
    main()

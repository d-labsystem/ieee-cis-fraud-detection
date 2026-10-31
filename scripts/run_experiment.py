"""Add feature steps, train XGBoost over several seeds, compare paired by seed.

Each (steps, seed) run appends one row to <processed_dir>/experiments.csv and
saves validation predictions to <processed_dir>/predictions/. The final table
compares every logged configuration against a reference, paired by seed:
a gain only counts if it beats the seed-to-seed noise.

Usage (on Kaggle, GPU):
    python scripts/run_experiment.py --device cuda                         # baseline
    python scripts/run_experiment.py --device cuda --steps client
    python scripts/run_experiment.py --device cuda --steps client entity --compare client
    python scripts/run_experiment.py --device cuda --steps client entity dnorm --compare client+entity

Steps always run in the order of STEPS below, whatever order they are given in.
"""

import argparse
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.baseline import TARGET, train_baseline
from fraud.config import get_settings
from fraud.features.client import add_client_features
from fraud.features.clock import add_hour
from fraud.features.dnorm import add_dnorm_features
from fraud.features.entity import add_entity_features
from fraud.features.missing import add_missing_features
from fraud.features.vreduce import drop_correlated_v
from fraud.metrics import score
from fraud.split import time_split

# Order matters: "client" and "missing" read raw D columns, which
# "dnorm_replace" drops.
STEPS = {
    "client": add_client_features,
    "entity": add_entity_features,
    "missing": add_missing_features,
    "dnorm": add_dnorm_features,
    "dnorm_replace": partial(add_dnorm_features, replace=True),
    "vreduce": drop_correlated_v,
    "hour": add_hour,
}


def segment_scores(valid: pd.DataFrame, prob: np.ndarray, n_prev: pd.Series) -> pd.DataFrame:
    """Validation scores for new clients (no earlier transaction) vs returning ones."""
    is_new = (n_prev.loc[valid.index] == 0).to_numpy()
    y = valid[TARGET].to_numpy()
    rows = {}
    for name, mask in (("new client", is_new), ("returning client", ~is_new)):
        if 0 < y[mask].sum() < mask.sum():
            rows[name] = score(y[mask], prob[mask]).model_dump()
    return pd.DataFrame(rows).T


def compare(log: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Mean scores per configuration, and the mean paired change vs ``reference``."""
    log = log.drop_duplicates(["steps", "seed"], keep="last")
    ref = log[log["steps"] == reference].set_index("seed")
    rows = []
    for steps, runs in log.groupby("steps", sort=False):
        runs = runs.set_index("seed")
        seeds = runs.index.intersection(ref.index)
        d_pr = runs.loc[seeds, "pr_auc"] - ref.loc[seeds, "pr_auc"]
        d_roc = runs.loc[seeds, "roc_auc"] - ref.loc[seeds, "roc_auc"]
        rows.append({
            "steps": steps,
            "seeds": len(runs),
            "roc_auc": runs["roc_auc"].mean(),
            "pr_auc": runs["pr_auc"].mean(),
            "pr_auc_sd": runs["pr_auc"].std(),
            f"d_roc vs {reference}": d_roc.mean() if len(seeds) else np.nan,
            f"d_pr vs {reference}": d_pr.mean() if len(seeds) else np.nan,
            "pr_wins": f"{(d_pr > 0).sum()}/{len(seeds)}",
        })
    return pd.DataFrame(rows).set_index("steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Parquet to train on (default: <processed_dir>/train_raw.parquet).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", nargs="*", choices=list(STEPS), default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="XGBoost random_state values (default: 0 1 2).")
    parser.add_argument("--compare", default="baseline",
                        help="Logged configuration to compare against, e.g. 'client'.")
    args = parser.parse_args()
    if {"dnorm", "dnorm_replace"} <= set(args.steps):
        parser.error("choose either dnorm or dnorm_replace, not both")
    steps_in_order = [s for s in STEPS if s in args.steps]

    settings = get_settings()
    path = args.path or settings.processed_dir / "train_raw.parquet"
    df = pd.read_parquet(path)
    raw_columns = set(df.columns)
    # Client history for the segment report only (not a feature unless the
    # "client" step adds it), so every run, the baseline too, gets the report.
    n_prev = add_client_features(df)["uid_n_prev"]

    # Features are built on the full table before splitting. That is safe only
    # because every step is causal (uses earlier rows only).
    for step in steps_in_order:
        start = time.perf_counter()
        df = STEPS[step](df)
        print(f"Step {step!r}: {time.perf_counter() - start:.1f}s")
    new_columns = [c for c in df.columns if c not in raw_columns]
    dropped = sorted(raw_columns - set(df.columns))
    if dropped:
        print(f"Dropped raw columns: {', '.join(dropped)}")
    split = time_split(df)
    steps = "+".join(steps_in_order) or "baseline"

    log_path = settings.processed_dir / "experiments.csv"
    pred_dir = settings.processed_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    segments, gains = [], []

    for seed in args.seeds:
        start = time.perf_counter()
        result = train_baseline(split, device=args.device, verbose=False, random_state=seed)
        s = result.valid_scores
        print(f"seed {seed}: ROC-AUC {s.roc_auc:.4f} | PR-AUC {s.pr_auc:.4f} | "
              f"best iteration {result.best_iteration} | {time.perf_counter() - start:.0f}s")

        segments.append(segment_scores(split.valid, result.valid_prob, n_prev))
        # total_gain, not "gain": XGBoost's "gain" is the average per split.
        gain = pd.Series(result.model.get_booster().get_score(importance_type="total_gain"))
        gains.append(pd.DataFrame({
            "gain_%": 100 * gain / gain.sum(),
            "rank": gain.rank(ascending=False),
        }).reindex(new_columns))

        pd.DataFrame({
            "TransactionID": split.valid["TransactionID"].to_numpy(),
            TARGET: split.valid[TARGET].to_numpy(),
            "prob": result.valid_prob,
        }).to_parquet(pred_dir / f"{steps}_seed{seed}.parquet", index=False)

        row = pd.DataFrame([{
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "steps": steps,
            "seed": seed,
            "n_features": len(result.features),
            "best_iteration": result.best_iteration,
            "roc_auc": round(s.roc_auc, 4),
            "pr_auc": round(s.pr_auc, 4),
            "pr_lift": round(s.pr_lift, 2),
        }])
        log = pd.concat([pd.read_csv(log_path), row]) if log_path.exists() else row
        log.to_csv(log_path, index=False)

    cols = ["n_transactions", "n_frauds", "roc_auc", "pr_auc", "pr_lift"]
    print(f"\nBy client history (mean over {len(args.seeds)} seeds):")
    print(pd.concat(segments).groupby(level=0)[cols].mean().to_string(float_format="{:.4f}".format))

    if new_columns:
        new = pd.concat(gains).groupby(level=0).mean().sort_values("gain_%", ascending=False)
        print(f"\nNew features, mean over seeds (share of total gain; rank out of "
              f"{len(result.features)}; NaN = never used). All new: {new['gain_%'].sum():.1f}%")
        print(new.to_string(float_format="{:.2f}".format))

    print(f"\nAll configurations ({log_path}):")
    print(compare(pd.read_csv(log_path), args.compare).to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()

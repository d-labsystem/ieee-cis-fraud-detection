"""Calibrate the final model's scores and use them for $-based alerting.

The validation month is split in time: the calibrator is fitted on the first
half and every number below is measured on the second half, which neither the
calibrator nor any threshold choice has seen.

Reads <processed_dir>/explain/valid_predictions.parquet (from explain.py) and
TransactionDT / TransactionAmt from the training parquet; no retraining.

Usage (on Kaggle, after explain.py):
    python scripts/calibrate.py --path /kaggle/working/train_raw.parquet
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud.baseline import TARGET
from fraud.calibration import (
    calibration_scores,
    expected_benefit,
    fit_isotonic,
    fit_platt,
    reliability_table,
    time_halves,
)
from fraud.config import get_settings
from fraud.threshold import best_alerts, capacity_table, cost_curve

CAPACITY_SHARES = (0.01, 0.02, 0.05)


def policy_row(y, rank_score, amt, k: int, review_cost: float, false_alarm_cost: float) -> dict:
    """Outcome of alerting on the k rows with the highest ``rank_score``."""
    top = np.argsort(-rank_score, kind="stable")[:k]
    fraud = y[top] == 1
    caught_amt = amt[top][fraud].sum()
    missed_amt = amt[y == 1].sum() - caught_amt
    return {
        "alerts": k,
        "share_%": 100 * k / len(y),
        "frauds_caught": int(fraud.sum()),
        "fraud_$_caught": caught_amt,
        "total_cost_$": review_cost * k + false_alarm_cost * (~fraud).sum() + missed_amt,
    }


def plot_reliability(y, curves: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([1e-4, 1], [1e-4, 1], color="grey", linestyle="--", label="perfect")
    for name, prob in curves.items():
        table = reliability_table(y, prob)
        ax.plot(table["mean_pred"], table["fraud_rate"].clip(lower=1e-4), marker="o", label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("actual fraud rate (bin)")
    ax.set_title("Reliability on the second half of the validation month")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Parquet with TransactionDT/TransactionAmt (default: <processed_dir>/train_raw.parquet).")
    parser.add_argument("--review-cost", type=float, default=3.0, help="$ per reviewed alert.")
    parser.add_argument("--false-alarm-cost", type=float, default=10.0,
                        help="Extra $ when the reviewed transaction was legitimate.")
    args = parser.parse_args()
    costs = (args.review_cost, args.false_alarm_cost)

    settings = get_settings()
    out_dir = settings.processed_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = args.path or settings.processed_dir / "train_raw.parquet"
    pred = pd.read_parquet(settings.processed_dir / "explain" / "valid_predictions.parquet")
    raw = (pd.read_parquet(path, columns=["TransactionID", "TransactionDT", "TransactionAmt"])
           .set_index("TransactionID").loc[pred["TransactionID"]])
    y, prob = pred[TARGET].to_numpy(), pred["prob"].to_numpy(dtype=float)
    dt, amt = raw["TransactionDT"].to_numpy(), raw["TransactionAmt"].to_numpy(dtype=float)

    fit, ev = time_halves(dt)
    print(f"Fit half:  {fit.sum():,} transactions, fraud rate {y[fit].mean():.2%}")
    print(f"Eval half: {ev.sum():,} transactions, fraud rate {y[ev].mean():.2%}")

    platt = fit_platt(y[fit], prob[fit])
    isotonic = fit_isotonic(y[fit], prob[fit])
    print(f"Platt: p = sigmoid({platt.slope:.3f} * logit(score) {platt.intercept:+.3f})")
    (out_dir / "platt.json").write_text(json.dumps(asdict(platt), indent=2))

    ye, ae = y[ev], amt[ev]
    calibrated = {"raw score": prob[ev], "platt": platt(prob[ev]), "isotonic": isotonic(prob[ev])}

    print("\nEval half, calibration (lower is better) and ranking (should not move):")
    print(pd.DataFrame({name: calibration_scores(ye, p) for name, p in calibrated.items()}).T
          .to_string(float_format="{:.4f}".format))

    for name in ["raw score", "platt"]:
        print(f"\nReliability, {name} (eval half):")
        print(reliability_table(ye, calibrated[name]).to_string(float_format="{:.4f}".format))

    print(f"\nWho gets reviewed at a fixed capacity: rank by score vs by expected $ (platt p x amount):")
    by_score = capacity_table(ye, calibrated["platt"], ae, shares=CAPACITY_SHARES)
    by_dollars = capacity_table(ye, calibrated["platt"] * ae, ae, shares=CAPACITY_SHARES)
    cols = ["recall", "precision", "amount_recall"]
    print(pd.concat({"by score": by_score[cols], "by expected $": by_dollars[cols]}, axis=1)
          .to_string(float_format="{:.3f}".format))

    print(f"\nAlert policies at ${args.review_cost:g}/review + ${args.false_alarm_cost:g}/false alarm "
          f"(eval half; fraud ${ae[ye == 1].sum():,.0f}):")
    fit_share = best_alerts(cost_curve(y[fit], prob[fit], amt[fit], *costs))["alerts"] / fit.sum()
    hindsight_k = int(best_alerts(cost_curve(ye, prob[ev], ae, *costs))["alerts"])
    policies = {
        "no model": policy_row(ye, prob[ev], ae, 0, *costs),
        "score, share picked on fit half": policy_row(ye, prob[ev], ae, round(fit_share * len(ye)), *costs),
        "score, best share in hindsight": policy_row(ye, prob[ev], ae, hindsight_k, *costs),
    }
    for name in ["raw score", "platt", "isotonic"]:
        benefit = expected_benefit(calibrated[name], ae, *costs)
        policies[f"expected $ > 0, {name}"] = policy_row(ye, benefit, ae, int((benefit > 0).sum()), *costs)
    print(pd.DataFrame(policies).T.to_string(float_format="{:,.1f}".format))

    plot_reliability(ye, calibrated, out_dir / "reliability.png")
    print(f"\nSaved platt.json and reliability.png to {out_dir}")


if __name__ == "__main__":
    main()

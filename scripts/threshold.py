"""Choose the alert threshold on validation: review capacity and $ cost.

Reads saved validation predictions (no retraining):
    <processed_dir>/explain/valid_predictions.parquet      from explain.py (seed 0)
    <processed_dir>/predictions/<FINAL_STEPS>_seed*.parquet  from run_experiment.py, if present

Usage (on Kaggle, after explain.py):
    python scripts/threshold.py --path /kaggle/working/train_raw.parquet
    python scripts/threshold.py --path ... --review-cost 5 --false-alarm-cost 20
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud.baseline import TARGET
from fraud.config import get_settings
from fraud.features.pipeline import FINAL_STEPS, steps_name
from fraud.threshold import SHARES, best_alerts, capacity_table, cost_curve

REVIEW_COSTS = (1, 3, 5, 10, 20)
FALSE_ALARM_COSTS = (0, 10, 25)


def load_predictions(processed_dir: Path) -> dict[str, pd.DataFrame]:
    preds = {"explain (seed 0)": pd.read_parquet(processed_dir / "explain" / "valid_predictions.parquet")}
    for path in sorted((processed_dir / "predictions").glob(f"{steps_name(FINAL_STEPS)}_seed*.parquet")):
        preds[path.stem.rsplit("_", 1)[1]] = pd.read_parquet(path)
    return preds


def sensitivity(y, prob, amt) -> pd.DataFrame:
    """Cost-optimal alert share for a grid of cost assumptions."""
    no_model = amt[y == 1].sum()
    rows = []
    for review in REVIEW_COSTS:
        for false_alarm in FALSE_ALARM_COSTS:
            best = best_alerts(cost_curve(y, prob, amt, review, false_alarm))
            rows.append({
                "review_$": review, "false_alarm_$": false_alarm,
                "best_share_%": 100 * best["alerts"] / len(y),
                "threshold": best["threshold"],
                "saved_vs_no_model_%": 100 * (1 - best["total"] / no_model),
            })
    return pd.DataFrame(rows).set_index(["review_$", "false_alarm_$"])


def plot(y, prob, amt, curve: pd.DataFrame, path: Path) -> None:
    shares = np.linspace(0.001, 0.10, 100)
    table = capacity_table(y, prob, amt, shares=shares)
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    for col in ["recall", "amount_recall", "precision"]:
        left.plot(100 * shares, table[col], label=col)
    left.set_xlabel("% of transactions reviewed")
    left.set_title("What a given review capacity buys")
    left.legend()
    left.grid(alpha=0.3)

    k = curve["alerts"] / len(y) * 100
    window = k <= 10
    for col in ["review", "false_alarm", "missed_fraud", "total"]:
        right.plot(k[window], curve.loc[window, col] / 1000, label=col)
    best = best_alerts(curve)
    right.axvline(100 * best["alerts"] / len(y), color="grey", linestyle="--", label="cheapest")
    right.set_xlabel("% of transactions reviewed")
    right.set_ylabel("cost on the validation month ($ thousands)")
    right.set_title("Total cost by alert volume")
    right.legend()
    right.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Parquet with TransactionAmt (default: <processed_dir>/train_raw.parquet).")
    # Assumptions, not data: a few minutes of analyst time per review, and the
    # support call / lost goodwill when a real customer's payment is blocked.
    parser.add_argument("--review-cost", type=float, default=3.0, help="$ per reviewed alert.")
    parser.add_argument("--false-alarm-cost", type=float, default=10.0,
                        help="Extra $ when the reviewed transaction was legitimate.")
    args = parser.parse_args()

    settings = get_settings()
    out_dir = settings.processed_dir / "threshold"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = args.path or settings.processed_dir / "train_raw.parquet"
    amounts = pd.read_parquet(path, columns=["TransactionID", "TransactionAmt"]).set_index("TransactionID")

    preds = load_predictions(settings.processed_dir)
    main_pred = preds["explain (seed 0)"]
    y = main_pred[TARGET].to_numpy()
    prob = main_pred["prob"].to_numpy()
    amt = amounts.loc[main_pred["TransactionID"], "TransactionAmt"].to_numpy(dtype=float)
    print(f"Validation: {len(y):,} transactions, {int(y.sum()):,} frauds, "
          f"${amt[y == 1].sum():,.0f} of fraud")

    print("\nReview capacity (seed-0 model from explain.py):")
    print(capacity_table(y, prob, amt).to_string(float_format="{:.3f}".format))

    curve = cost_curve(y, prob, amt, args.review_cost, args.false_alarm_cost)
    best = best_alerts(curve)
    no_model = amt[y == 1].sum()
    print(f"\nCheapest at ${args.review_cost:g}/review + ${args.false_alarm_cost:g}/false alarm: "
          f"{int(best['alerts']):,} alerts ({100 * best['alerts'] / len(y):.2f}% of traffic), "
          f"threshold {best['threshold']:.3f}")
    print(f"  cost ${best['total']:,.0f} = review ${best['review']:,.0f} + false alarms "
          f"${best['false_alarm']:,.0f} + missed fraud ${best['missed_fraud']:,.0f}")
    print(f"  vs ${no_model:,.0f} with no model ({100 * (1 - best['total'] / no_model):.1f}% saved)")

    print("\nHow the cheapest alert share depends on the cost assumptions:")
    print(sensitivity(y, prob, amt).to_string(float_format="{:.3f}".format))

    print("\nScore threshold for a fixed review share, per model:")
    if len(preds) == 1:
        print("  only one model found; run run_experiment.py with the final steps to compare seeds")
    else:
        thresholds = pd.DataFrame({
            name: capacity_table(p[TARGET].to_numpy(), p["prob"].to_numpy(),
                                 amounts.loc[p["TransactionID"], "TransactionAmt"].to_numpy(float))["threshold"]
            for name, p in preds.items()
        })
        print(thresholds.to_string(float_format="{:.3f}".format))

    plot(y, prob, amt, curve, out_dir / "threshold.png")
    print(f"\nPlot saved to {out_dir / 'threshold.png'}")


if __name__ == "__main__":
    main()

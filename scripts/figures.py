"""README figures from the one-time test scoring (reads final.py's outputs).

    decay.png        weekly PR-AUC on the test months: final vs one-month-staler
                     proxy, and new clients vs clients already seen with fraud
    cost.png         total cost against share of traffic reviewed, with the
                     review share chosen on month 4 and the expected-$ rule
    reliability.png  predicted vs actual fraud rate, raw scores vs Platt

Makes no decision: every number drawn was already printed by final.py.

Usage (on Kaggle, after final.py):
    python scripts/figures.py --path /kaggle/working/train_raw.parquet
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud.baseline import TARGET
from fraud.calibration import reliability_table, expected_benefit
from fraud.config import get_settings
from fraud.drift import client_segments, segment_week_scores, week_scores
from fraud.features.pipeline import build_features
from fraud.split import TEST_MONTHS, TRAIN_MONTHS, VALID_MONTHS
from fraud.threshold import cost_curve, policy_outcome

# Categorical slots 1-2 of the reference palette (validated as a pair), plus ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
})


def _week_labels(weeks: pd.DataFrame) -> list[str]:
    """'wk 1' ... with the last, partial week marked by its length in days."""
    labels = [f"wk {w + 1}" for w in weeks.index]
    days = weeks["transactions"].iloc[-1] / weeks["transactions"].iloc[:-1].mean() * 7
    if days < 6:
        labels[-1] += f"\n(~{days:.0f} days)"
    return labels


def decay_figure(y, prob, proxy_prob, day, segment, path: Path) -> None:
    final_w, proxy_w = week_scores(y, prob, day), week_scores(y, proxy_prob, day)
    seg = segment_week_scores(y, prob, day, segment)["pr_auc"].unstack("segment")
    x = np.arange(len(final_w))
    labels = _week_labels(final_w)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, series, title in [
        (left, [("trained on months 0-4", final_w["pr_auc"], BLUE),
                ("trained on months 0-3 (one month staler)", proxy_w["pr_auc"], ORANGE)],
         "Fresher labels: final model vs proxy"),
        (right, [("new clients", seg["new client"], BLUE),
                 ("clients already seen with fraud", seg["known, fraud"], ORANGE)],
         "By client (final model)"),
    ]:
        for name, values, color in series:
            ax.plot(x, values.to_numpy(), marker="o", color=color, label=name)
            ax.annotate(f"{values.iloc[-1]:.2f}", (x[-1], values.iloc[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", color=INK)
        ax.set_title(title, loc="left")
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.3, len(x) - 0.4)
        ax.legend(loc="lower left")
    left.set_ylabel("PR-AUC (average precision)")
    left.set_ylim(0, 1)
    fig.suptitle("Test months 5-6, week by week", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_figure(y, prob, platt_p, amt, policy: dict, path: Path) -> None:
    costs = policy["review_cost"], policy["false_alarm_cost"]
    curve = cost_curve(y, prob, amt, *costs)
    share = 100 * curve["alerts"] / len(y)
    keep = share <= 30
    k_share = round(policy["review_share"] * len(y))
    benefit = expected_benefit(platt_p, amt, *costs)
    by_share = policy_outcome(y, prob, amt, k_share, *costs)
    by_dollar = policy_outcome(y, benefit, amt, int((benefit > 0).sum()), *costs)
    no_model = curve["total"].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(share[keep], curve["total"][keep] / 1e3, color=BLUE, label="review the top-scored share")
    ax.axhline(no_model / 1e3, color=MUTED, linestyle="--", linewidth=1.2)
    ax.annotate(f"no model: \\${no_model / 1e3:.0f}k of fraud lost", (1, no_model / 1e3),
                xytext=(0, 6), textcoords="offset points", ha="left", color=MUTED)
    for p, color, text, offset in [
        (by_share, BLUE, f"month-4 review share ({by_share['share_%']:.1f}%)", (10, 40)),
        (by_dollar, ORANGE, f"expected \\$ > 0, Platt ({by_dollar['share_%']:.1f}%)", (12, -18)),
    ]:
        ax.scatter(p["share_%"], p["total_cost_$"] / 1e3, s=70, color=color, zorder=3,
                   edgecolor="#fcfcfb", linewidth=2)
        saved = 1 - p["total_cost_$"] / no_model
        ax.annotate(f"{text}\n\\${p['total_cost_$'] / 1e3:.0f}k, saves {saved:.0%}",
                    (p["share_%"], p["total_cost_$"] / 1e3), xytext=offset,
                    textcoords="offset points", color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax.set_xlabel("share of transactions reviewed (%)")
    ax.set_ylabel("total cost (\\$k)")
    ax.set_ylim(0, no_model / 1e3 * 1.1)
    ax.set_title(f"Cost of fraud + reviews on the test months (\\${costs[0]:g}/review, "
                 f"\\${costs[1]:g}/false alarm)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability_figure(y, prob, platt_p, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    lims = (1e-4, 1)
    ax.plot(lims, lims, color=MUTED, linestyle="--", linewidth=1.2, label="perfect calibration")
    for name, p, color in [("raw scores", prob, ORANGE), ("after Platt scaling", platt_p, BLUE)]:
        table = reliability_table(y, p)
        table = table[(table["transactions"] >= 50) & (table["fraud_rate"] > 0)]
        ax.plot(table["mean_pred"], table["fraud_rate"], marker="o", color=color, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("predicted fraud probability (bin mean)")
    ax.set_ylabel("actual fraud rate")
    ax.set_title("Calibration on the test months", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Raw parquet (default: <processed_dir>/train_raw.parquet).")
    parser.add_argument("--out", type=Path, default=None, help="Default: <processed_dir>/figures.")
    args = parser.parse_args()

    settings = get_settings()
    final_dir = settings.processed_dir / "final"
    out_dir = args.out or settings.processed_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only the client step: segments need its D1n, nothing else is needed.
    df = build_features(pd.read_parquet(args.path or settings.processed_dir / "train_raw.parquet"),
                        ["client"])
    segment = client_segments(df, TRAIN_MONTHS + VALID_MONTHS, TEST_MONTHS)
    test = df.loc[segment.index]
    pred = (pd.read_parquet(final_dir / "test_predictions.parquet")
            .set_index("TransactionID").loc[test["TransactionID"]])
    policy = json.loads((final_dir / "policy.json").read_text())

    y, amt, day = test[TARGET].to_numpy(), test["TransactionAmt"].to_numpy(float), test["day"].to_numpy()
    prob, platt_p, proxy_prob = (pred[c].to_numpy() for c in ("prob", "platt_p", "proxy_prob"))

    decay_figure(y, prob, proxy_prob, day, segment.to_numpy(), out_dir / "decay.png")
    cost_figure(y, prob, platt_p, amt, policy, out_dir / "cost.png")
    reliability_figure(y, prob, platt_p, out_dir / "reliability.png")
    print(f"Wrote decay.png, cost.png, reliability.png to {out_dir}")


if __name__ == "__main__":
    main()

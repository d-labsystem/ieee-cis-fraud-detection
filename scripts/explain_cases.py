"""Explain single transactions: why was this one flagged (or missed)?

Reads what scripts/explain.py saved in <processed_dir>/explain/ (no retraining)
and writes per-case waterfall plots there.

Usage (on Kaggle, after explain.py):
    python scripts/explain_cases.py --path /kaggle/working/train_raw.parquet
    python scripts/explain_cases.py --path ... --ids 3456789 3456790   # your own picks
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap as shap_lib

from fraud.baseline import TARGET
from fraud.config import get_settings
from fraud.explain import BIAS, category_effects, pick_cases, reasons
from fraud.features.pipeline import FINAL_STEPS, build_features
from fraud.split import time_split

CATEGORICAL = ["ProductCD", "P_emaildomain", "R_emaildomain", "card6", "id_31", "browser", "os",
               "device_brand", "DeviceType"]
WATERFALL_FEATURES = 12


def waterfall(shap_row: pd.Series, x_row: pd.Series, title: str, path: Path) -> None:
    features = [c for c in shap_row.index if c != BIAS]
    values = [("missing" if pd.isna(x_row[f]) else x_row[f]) for f in features]
    explanation = shap_lib.Explanation(
        values=shap_row[features].to_numpy(dtype=float),
        base_values=float(shap_row[BIAS]),
        data=np.array(values, dtype=object),
        feature_names=features,
    )
    shap_lib.plots.waterfall(explanation, max_display=WATERFALL_FEATURES, show=False)
    fig = plt.gcf()
    fig.suptitle(title)
    fig.set_size_inches(10, 7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def explain_cases(
    valid: pd.DataFrame,
    shap: pd.DataFrame,
    pred: pd.DataFrame,
    features: list[str],
    out_dir: Path,
    alert_share: float,
    ids: list[int],
) -> None:
    """``valid``, ``shap`` and ``pred`` are aligned row by row."""
    X, y = valid[features], valid[TARGET]

    print("Mean push per category (read categorical features here, not in the beeswarm):")
    for col in [c for c in CATEGORICAL if c in features]:
        table = category_effects(shap, X, y, col, top=8)
        print(f"\n{col}:")
        print(table.to_string(float_format="{:.3f}".format))

    prob = pred["prob"].to_numpy()
    threshold = np.quantile(prob, 1 - alert_share)
    alerts = prob >= threshold
    caught = (alerts & (y.to_numpy() == 1)).sum()
    print(f"\nIf analysts review the top {alert_share:.0%} of scores "
          f"({alerts.sum():,} alerts, threshold {threshold:.3f}):")
    print(f"  {caught:,} of {int(y.sum()):,} frauds caught (recall {caught / y.sum():.1%}), "
          f"precision {caught / alerts.sum():.1%}")

    cases = pick_cases(pred, alert_share)
    id_to_pos = pd.Series(np.arange(len(valid)), index=valid["TransactionID"].to_numpy())
    for tid in ids:
        if tid not in id_to_pos.index:
            raise ValueError(f"TransactionID {tid} is not in the validation month.")
        cases[f"transaction {tid}"] = int(id_to_pos[tid])

    for name, pos in cases.items():
        tid = int(valid["TransactionID"].iloc[pos])
        p, is_fraud = prob[pos], int(y.iloc[pos])
        pct = 100 * (prob > p).mean()
        print(f"\n=== {name}: TransactionID {tid} | actual {'FRAUD' if is_fraud else 'legit'} | "
              f"score {p:.3f} (top {pct:.2f}% of scores) | "
              f"{'ALERT' if p >= threshold else 'no alert'} ===")
        print(f"start from the average: {shap[BIAS].iloc[pos]:+.2f} log-odds, then:")
        print(reasons(shap.iloc[pos], X.iloc[pos]).to_string(float_format="{:+.3f}".format))
        path = out_dir / f"case_{name.replace(' ', '_')}.png"
        waterfall(shap.iloc[pos], X.iloc[pos], f"{name}: {tid} (score {p:.3f})", path)
    print(f"\nWaterfall plots saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Parquet the model was trained on (default: <processed_dir>/train_raw.parquet).")
    parser.add_argument("--alert-share", type=float, default=0.02,
                        help="Share of transactions analysts review (default 0.02 = top 2%%).")
    parser.add_argument("--ids", type=int, nargs="*", default=[],
                        help="Extra validation TransactionIDs to explain.")
    args = parser.parse_args()

    settings = get_settings()
    out_dir = settings.processed_dir / "explain"
    meta = json.loads((out_dir / "meta.json").read_text())
    if meta["steps"] != list(FINAL_STEPS):
        raise ValueError(f"Saved model used steps {meta['steps']}, current FINAL_STEPS is "
                         f"{list(FINAL_STEPS)}: rerun scripts/explain.py.")

    path = args.path or settings.processed_dir / "train_raw.parquet"
    valid = time_split(build_features(pd.read_parquet(path), FINAL_STEPS)).valid
    shap = pd.read_parquet(out_dir / "shap_valid.parquet")
    pred = pd.read_parquet(out_dir / "valid_predictions.parquet")
    # Align everything on TransactionID rather than trusting row order.
    order = pd.Index(valid["TransactionID"])
    shap = shap.set_index("TransactionID").loc[order].reset_index(drop=True)
    pred = pred.set_index("TransactionID").loc[order].reset_index()
    valid = valid.reset_index(drop=True)
    explain_cases(valid, shap, pred, meta["features"], out_dir, args.alert_share, args.ids)


if __name__ == "__main__":
    main()

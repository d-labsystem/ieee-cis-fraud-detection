"""Train the final Phase 4 model and explain it with SHAP on the validation month.

Writes to <processed_dir>/explain/:
    model.json, meta.json         the model and what it was trained with
    shap_valid.parquet            SHAP values per validation transaction
    valid_predictions.parquet     TransactionID, isFraud, prob
    importance.csv                global importance per feature
    beeswarm.png, dependence.png  plots

Usage (on Kaggle, GPU):
    python scripts/explain.py --path /kaggle/working/train_raw.parquet --device cuda
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

from fraud.baseline import TARGET, train_baseline
from fraud.config import get_settings
from fraud.explain import BIAS, family_importance, global_importance, shap_values
from fraud.features.pipeline import FINAL_STEPS, build_features, steps_name
from fraud.split import Split, time_split

PLOT_ROWS = 10_000  # beeswarm/dependence sample; all rows are used for the numbers
DEPENDENCE = ["hour", "D1n", "card1_share_30d", "TransactionAmt", "uid_amt_ratio", "D15n"]


def numeric_view(X: pd.DataFrame) -> pd.DataFrame:
    """Categories -> integer codes (NaN kept), so plots can colour by value."""
    out = X.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].cat.codes.replace(-1, np.nan).astype(np.float32)
    return out


def plot_dependence(shap: pd.DataFrame, X: pd.DataFrame, features: list[str], path: Path) -> None:
    features = [f for f in features if f in X]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), features):
        x, s = X[col].astype(np.float64), shap[col]
        ax.scatter(x, s, s=2, alpha=0.3)
        if col == "TransactionAmt":
            ax.set_xscale("log")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel(col)
        ax.set_ylabel("SHAP (log-odds)")
        nan_push = s[x.isna()].mean()
        ax.set_title(f"{col}" + ("" if np.isnan(nan_push) else f"  (NaN: {nan_push:+.2f})"))
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)
    fig.suptitle("How the SHAP value changes with the feature value (validation sample)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def explain(split: Split, device: str, seed: int, out_dir: Path, top: int = 25) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_baseline(split, device=device, verbose=500, random_state=seed)
    s = result.valid_scores
    print(f"\nValid ROC-AUC {s.roc_auc:.4f} | PR-AUC {s.pr_auc:.4f} | best iteration {result.best_iteration}")

    result.model.save_model(out_dir / "model.json")
    (out_dir / "meta.json").write_text(json.dumps({
        "steps": list(FINAL_STEPS), "seed": seed, "best_iteration": result.best_iteration,
        "features": result.features, "valid_scores": s.model_dump(),
    }, indent=2))
    pd.DataFrame({
        "TransactionID": split.valid["TransactionID"].to_numpy(),
        TARGET: split.valid[TARGET].to_numpy(),
        "prob": result.valid_prob,
    }).to_parquet(out_dir / "valid_predictions.parquet", index=False)

    X = split.valid[result.features]
    shap = shap_values(result.model, X, result.features, result.best_iteration)
    shap.assign(TransactionID=split.valid["TransactionID"].to_numpy()).to_parquet(
        out_dir / "shap_valid.parquet", index=False
    )
    print(f"Average prediction (bias): {shap[BIAS].iloc[0]:+.3f} log-odds "
          f"= {1 / (1 + np.exp(-shap[BIAS].iloc[0])):.2%}")

    imp = global_importance(shap, X)
    gain = pd.Series(result.model.get_booster().get_score(importance_type="total_gain"))
    imp["gain_rank"] = gain.rank(ascending=False).reindex(imp.index)
    imp.to_csv(out_dir / "importance.csv")

    cols = ["family", "share_%", "direction", "missing", "shap_if_missing", "gain_rank"]
    print(f"\nTop {top} features by mean |SHAP| (direction: +1 = higher value -> more fraud):")
    print(imp[cols].head(top).to_string(float_format="{:.3f}".format))
    print("\nShare of the model's explanation by feature family:")
    print(family_importance(imp).to_string(float_format="{:.1f}".format))

    sample = np.random.default_rng(seed).choice(len(X), size=min(PLOT_ROWS, len(X)), replace=False)
    Xs = numeric_view(X.iloc[sample])
    shap_lib.summary_plot(
        shap.iloc[sample][result.features].to_numpy(), Xs, max_display=20, show=False
    )
    plt.title("Top 20 features: each dot is one validation transaction")
    plt.tight_layout()
    plt.savefig(out_dir / "beeswarm.png", dpi=110)
    plt.close()
    plot_dependence(shap.iloc[sample], Xs, DEPENDENCE, out_dir / "dependence.png")
    print(f"\nSaved to {out_dir}")
    return imp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Parquet to train on (default: <processed_dir>/train_raw.parquet).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings()
    path = args.path or settings.processed_dir / "train_raw.parquet"
    df = build_features(pd.read_parquet(path), FINAL_STEPS, verbose=True)
    print(f"Configuration: {steps_name(FINAL_STEPS)}")
    explain(time_split(df), args.device, args.seed, settings.processed_dir / "explain")


if __name__ == "__main__":
    main()

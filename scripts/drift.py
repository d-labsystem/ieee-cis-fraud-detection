"""Phase 7b/7c: what drifts between training and the test months, and does it matter?

No test-month labels are read: the label column is dropped from the test rows
before anything else happens. Uses the model saved by explain.py and the
Platt calibrator saved by calibrate.py; no retraining except the small
adversarial-validation classifiers.

Usage (on Kaggle, after explain.py and calibrate.py):
    python scripts/drift.py --path /kaggle/working/train_raw.parquet --device cuda
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from fraud.baseline import TARGET
from fraud.calibration import Platt, expected_benefit
from fraud.config import get_settings
from fraud.drift import adversarial_validation, psi_table, week_scores
from fraud.features.pipeline import FINAL_STEPS, build_features
from fraud.split import time_split

TOP = 20
LARGE_PSI = 0.25


def model_scores(model_dir: Path, X: pd.DataFrame, device: str) -> np.ndarray:
    meta = json.loads((model_dir / "meta.json").read_text())
    booster = xgb.Booster()
    booster.load_model(model_dir / "model.json")
    booster.set_param({"device": device})
    dmatrix = xgb.DMatrix(X[meta["features"]], enable_categorical=True)
    return booster.predict(dmatrix, iteration_range=(0, meta["best_iteration"] + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Raw parquet (default: <processed_dir>/train_raw.parquet).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--review-cost", type=float, default=3.0)
    parser.add_argument("--false-alarm-cost", type=float, default=10.0)
    args = parser.parse_args()

    settings = get_settings()
    model_dir = settings.processed_dir / "explain"
    meta = json.loads((model_dir / "meta.json").read_text())
    if meta["steps"] != list(FINAL_STEPS):
        raise ValueError("Saved model is not the FINAL_STEPS model: rerun scripts/explain.py.")
    features = meta["features"]
    shap_share = pd.read_csv(model_dir / "importance.csv", index_col=0)["share_%"]

    path = args.path or settings.processed_dir / "train_raw.parquet"
    split = time_split(build_features(pd.read_parquet(path), FINAL_STEPS))
    train, valid = split.train, split.valid
    test = split.test.drop(columns=TARGET)  # never look at test labels before the final scoring
    print(f"Train months 0-3: {len(train):,} | valid month 4: {len(valid):,} | "
          f"test months 5-6: {len(test):,} rows (labels dropped)")

    # ---------- 7b: feature drift ----------
    psi_valid = psi_table(train, valid, features)
    psi_test = psi_table(train, test, features)
    table = pd.DataFrame({"psi_valid": psi_valid, "psi_test": psi_test,
                          "model_shap_%": shap_share}).reindex(psi_test.index)
    print(f"\nPSI vs training months, top {TOP} by test drift (> {LARGE_PSI} = large shift):")
    print(table.head(TOP).to_string(float_format="{:.3f}".format))
    for name, col in [("valid", "psi_valid"), ("test", "psi_test")]:
        large = table[col] > LARGE_PSI
        print(f"{name}: {large.sum()} features with PSI > {LARGE_PSI}, carrying "
              f"{table.loc[large, 'model_shap_%'].sum():.1f}% of the model's SHAP")
    print("\nMost-used model features and their drift:")
    print(table.sort_values("model_shap_%", ascending=False).head(10)
          .to_string(float_format="{:.3f}".format))

    print("\nAdversarial validation (can a classifier tell the periods apart?):")
    for name, current in [("train vs valid", valid), ("train vs test", test)]:
        auc, gain = adversarial_validation(train, current, features, device=args.device)
        print(f"\n{name}: AUC {auc:.3f}")
        top = pd.DataFrame({"gain_%": gain, "psi_test": psi_test,
                            "model_shap_%": shap_share}).reindex(gain.index).head(10)
        print(top.to_string(float_format="{:.3f}".format))

    # ---------- 7b: score drift ----------
    platt = Platt(**json.loads((settings.processed_dir / "calibration" / "platt.json").read_text()))
    valid_score = model_scores(model_dir, valid, args.device)
    alert_cut = np.quantile(valid_score, 0.98)  # top 2% of the validation month
    rows = {}
    for name, part in [("valid (4)", valid), *[(f"test ({m})", test[test["month"] == m])
                                                  for m in sorted(test["month"].unique())]]:
        raw = valid_score if part is valid else model_scores(model_dir, part, args.device)
        p = platt(raw)
        benefit = expected_benefit(p, part["TransactionAmt"].to_numpy(float),
                                   args.review_cost, args.false_alarm_cost)
        rows[name] = {"transactions": len(part), "mean_raw": raw.mean(), "mean_platt_p": p.mean(),
                      "share_above_valid_top2%_cut": (raw >= alert_cut).mean(),
                      "share_expected_$>0": (benefit > 0).mean()}
    print("\nModel output by month (no labels; mean_platt_p is the fraud rate the model expects):")
    print(pd.DataFrame(rows).T.to_string(float_format="{:.4f}".format))

    # ---------- 7c: weekly decay on the validation month ----------
    print("\nValidation month week by week (same model):")
    weekly = week_scores(valid[TARGET].to_numpy(), valid_score, valid["day"].to_numpy())
    print(weekly.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()

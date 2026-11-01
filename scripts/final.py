"""Final step: train the final model and score the test months ONCE.

Every decision is fixed before a test label is read:

1. proxy model: the final recipe trained on months 0-3 (fixed tree count,
   3-seed ensemble). Its month-4 predictions fit the Platt calibrator and
   pick the cost-optimal review share.
2. final model: the same recipe on months 0-4, the freshest labels (Phase 7:
   a one-month label delay cost 15% PR-AUC).
3. test months 5-6 scored with the final model + Platt. The proxy model is
   scored too, only to report what the fresher month was worth.

Writes <processed_dir>/final/ (test predictions, platt.json) and refuses to
run again once scored there unless --force: the point is to look once.

Usage (on Kaggle, GPU, ~25 min):
    python scripts/final.py --path /kaggle/working/train_raw.parquet --device cuda
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.baseline import TARGET, feature_columns
from fraud.calibration import calibration_scores, expected_benefit, fit_platt
from fraud.config import get_settings
from fraud.drift import SEGMENTS, client_segments, segment_week_scores, week_scores
from fraud.features.pipeline import FINAL_STEPS, build_features, steps_name
from fraud.final import N_TREES, SEEDS, ensemble_predict, train_ensemble
from fraud.metrics import score
from fraud.split import TEST_MONTHS, TRAIN_MONTHS, VALID_MONTHS, time_split
from fraud.threshold import best_alerts, capacity_table, cost_curve


def policy(y, rank_score, amt, k: int, review_cost: float, false_alarm_cost: float) -> dict:
    top = np.argsort(-rank_score, kind="stable")[:k]
    fraud = y[top] == 1
    caught = amt[top][fraud].sum()
    return {"alerts": k, "share_%": 100 * k / len(y), "frauds_caught": int(fraud.sum()),
            "recall": fraud.sum() / y.sum(), "fraud_$_caught": caught,
            "total_cost_$": review_cost * k + false_alarm_cost * (~fraud).sum() + amt[y == 1].sum() - caught}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Raw parquet (default: <processed_dir>/train_raw.parquet).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--trees", type=int, default=N_TREES)
    parser.add_argument("--review-cost", type=float, default=3.0)
    parser.add_argument("--false-alarm-cost", type=float, default=10.0)
    parser.add_argument("--force", action="store_true", help="Score again even if already scored.")
    args = parser.parse_args()
    costs = (args.review_cost, args.false_alarm_cost)

    settings = get_settings()
    out_dir = settings.processed_dir / "final"
    marker = out_dir / "SCORED"
    if marker.exists() and not args.force:
        raise SystemExit(f"Test months already scored ({marker}). Results are in {out_dir}.")
    out_dir.mkdir(parents=True, exist_ok=True)

    path = args.path or settings.processed_dir / "train_raw.parquet"
    df = build_features(pd.read_parquet(path), FINAL_STEPS)
    split = time_split(df)
    features = feature_columns(split.train)
    print(f"Features: {steps_name(FINAL_STEPS)} ({len(features)}) | {args.trees} trees x seeds {SEEDS}")

    # ---------- 1. proxy model (months 0-3): calibration + review share ----------
    proxy = train_ensemble(split.train, features, args.trees, args.device, SEEDS)
    valid_prob = ensemble_predict(proxy, split.valid, features)
    y_valid = split.valid[TARGET].to_numpy()
    valid_amt = split.valid["TransactionAmt"].to_numpy(float)
    print(f"Proxy on month 4: ROC-AUC {score(y_valid, valid_prob).roc_auc:.4f} | "
          f"PR-AUC {score(y_valid, valid_prob).pr_auc:.4f}")
    platt = fit_platt(y_valid, valid_prob)
    (out_dir / "platt.json").write_text(json.dumps(asdict(platt), indent=2))
    review_share = best_alerts(cost_curve(y_valid, valid_prob, valid_amt, *costs))["alerts"] / len(y_valid)
    print(f"Platt: slope {platt.slope:.3f}, intercept {platt.intercept:+.3f} | "
          f"cost-optimal review share on month 4: {review_share:.2%}")

    # ---------- 2. final model (months 0-4) ----------
    train_04 = pd.concat([split.train, split.valid])
    final = train_ensemble(train_04, features, args.trees, args.device, SEEDS)

    # ---------- 3. test months: predictions first, labels last ----------
    test = split.test
    test_prob = ensemble_predict(final, test, features)
    proxy_test_prob = ensemble_predict(proxy, test, features)
    test_p = platt(test_prob)
    pd.DataFrame({"TransactionID": test["TransactionID"].to_numpy(), "prob": test_prob,
                  "platt_p": test_p, "proxy_prob": proxy_test_prob}).to_parquet(
        out_dir / "test_predictions.parquet", index=False)
    marker.write_text("Test months scored once; rerun only with --force.\n")

    y = test[TARGET].to_numpy()  # the first and only read of test labels
    amt = test["TransactionAmt"].to_numpy(float)
    print(f"\n===== TEST months {TEST_MONTHS}: {len(y):,} transactions, {int(y.sum()):,} frauds "
          f"({y.mean():.2%}), ${amt[y == 1].sum():,.0f} of fraud =====")

    overall = pd.DataFrame({
        "final (months 0-4)": score(y, test_prob).model_dump(),
        "proxy (months 0-3)": score(y, proxy_test_prob).model_dump(),
    }).T[["roc_auc", "pr_auc", "pr_lift"]]
    print("\nRanking on the test months:")
    print(overall.to_string(float_format="{:.4f}".format))

    print("\nCalibration of the final model (Platt fitted on the proxy's month 4):")
    print(pd.DataFrame({"raw": calibration_scores(y, test_prob), "platt": calibration_scores(y, test_p)})
          .T[["brier", "log_loss", "ece", "mean_pred", "fraud_rate"]].to_string(float_format="{:.4f}".format))

    segment = client_segments(df, TRAIN_MONTHS + VALID_MONTHS, TEST_MONTHS).loc[test.index].to_numpy()
    rows = {}
    for name in SEGMENTS:
        m = segment == name
        s = score(y[m], test_prob[m]) if 0 < y[m].sum() < m.sum() else None
        rows[name] = {"share_of_rows": m.mean(), "share_of_frauds": y[m].sum() / y.sum(),
                      "fraud_rate": y[m].mean(), "roc_auc": s.roc_auc if s else np.nan,
                      "pr_auc": s.pr_auc if s else np.nan}
    print("\nBy client (known = seen in months 0-4):")
    print(pd.DataFrame(rows).T.to_string(float_format="{:.4f}".format))

    day = test["day"].to_numpy()
    print("\nWeek by week (final model):")
    print(week_scores(y, test_prob, day)[["first_day", "transactions", "fraud_rate", "roc_auc", "pr_auc"]]
          .to_string(float_format="{:.4f}".format))
    short = dict(zip(SEGMENTS, ["new", "known", "known+fraud"]))
    weekly = segment_week_scores(y, test_prob, day, pd.Series(segment).map(short).to_numpy())
    print("\nWeek by week, PR-AUC by client segment:")
    print(weekly["pr_auc"].unstack("segment").reindex(columns=list(short.values()))
          .to_string(float_format="{:.4f}".format))

    print("\nReview capacity on the test months (final model):")
    print(capacity_table(y, test_prob, amt).drop(columns="threshold").to_string(float_format="{:.3f}".format))

    benefit = expected_benefit(test_p, amt, *costs)
    policies = {
        "no model": policy(y, test_prob, amt, 0, *costs),
        f"score, review share from month 4 ({review_share:.1%})":
            policy(y, test_prob, amt, round(review_share * len(y)), *costs),
        "expected $ > 0 (Platt)": policy(y, benefit, amt, int((benefit > 0).sum()), *costs),
    }
    print(f"\nAlert policies at ${args.review_cost:g}/review + ${args.false_alarm_cost:g}/false alarm:")
    print(pd.DataFrame(policies).T.to_string(float_format="{:,.3f}".format))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

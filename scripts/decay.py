"""Phase 7d: why does the model weaken week by week through the validation month?

Hypothesis: much of its skill is recognising clients it saw in training
(IEEE-CIS labels a client's later transactions as fraud once one is reported),
and each week a smaller share of traffic comes from those clients.

Splits validation rows by what training knew about their client (uid from
fraud.features.client: card1 + addr1 + D1n), using training-month labels only:

    new client      - no transaction in months 0-3
    known, no fraud - seen in training, never labelled fraud
    known, fraud    - seen in training with at least one fraud

Reads the model's validation predictions from explain.py; no retraining.

Usage (on Kaggle, after explain.py):
    python scripts/decay.py --path /kaggle/working/train_raw.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.baseline import TARGET
from fraud.config import get_settings
from fraud.drift import SEGMENTS, client_segments, segment_week_scores
from fraud.features.pipeline import FINAL_STEPS, build_features
from fraud.metrics import score
from fraud.split import TRAIN_MONTHS, VALID_MONTHS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=None,
                        help="Raw parquet (default: <processed_dir>/train_raw.parquet).")
    args = parser.parse_args()

    settings = get_settings()
    path = args.path or settings.processed_dir / "train_raw.parquet"
    df = build_features(pd.read_parquet(path), FINAL_STEPS)
    segment = client_segments(df, TRAIN_MONTHS, VALID_MONTHS)
    valid = df.loc[segment.index]
    pred = (pd.read_parquet(settings.processed_dir / "explain" / "valid_predictions.parquet")
            .set_index("TransactionID").loc[valid["TransactionID"]])
    y, prob = valid[TARGET].to_numpy(), pred["prob"].to_numpy()

    print("Validation month by client segment (whole month):")
    rows = {}
    for name in SEGMENTS:
        mask = (segment == name).to_numpy()
        frauds = int(y[mask].sum())
        # Scores need both classes; a segment can lack one on small data.
        s = score(y[mask], prob[mask]) if 0 < frauds < mask.sum() else None
        rows[name] = {"transactions": int(mask.sum()), "share_of_rows": mask.mean(),
                      "frauds": frauds, "share_of_frauds": frauds / y.sum(),
                      "fraud_rate": y[mask].mean() if mask.any() else np.nan,
                      "roc_auc": s.roc_auc if s else np.nan, "pr_auc": s.pr_auc if s else np.nan}
    print(pd.DataFrame(rows).T.to_string(float_format="{:.4f}".format))

    short = dict(zip(SEGMENTS, ["new", "known", "known+fraud"]))
    table = segment_week_scores(y, prob, valid["day"].to_numpy(), segment.map(short).to_numpy())
    for col in ["share_of_week", "fraud_rate", "roc_auc", "pr_auc"]:
        print(f"\nWeek by week, {col}:")
        wide = table[col].unstack("segment").reindex(columns=list(short.values()))
        print(wide.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()

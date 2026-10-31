"""Baseline: XGBoost on the raw columns, no feature engineering.

Every later phase is judged by how much it beats this. Early stopping uses
the validation month; the test month is not touched here.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from fraud.metrics import Scores, score
from fraud.split import Split

TARGET = "isFraud"

# Not features. TransactionID and TransactionDT (and day/month derived from
# it) only grow with time: every later transaction lies beyond the training
# range, so a tree can't use them on new data, and on training data they let
# it learn "early = less fraud" from the holiday month.
NON_FEATURES: tuple[str, ...] = (TARGET, "TransactionID", "TransactionDT", "day", "month")

PARAMS: dict = {
    # Upper bound; early stopping picks the real number. Raised from 2000 when
    # dnorm_replace runs hit it. Runs that stopped earlier are unaffected.
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "tree_method": "hist",
    "enable_categorical": True,  # uses the category dtypes from downcast()
    # Fraud review cares about precision among the top alerts, so stop on
    # PR-AUC. No scale_pos_weight: it would distort the probabilities that
    # Phase 5 turns into a decision threshold, and ranking barely needs it.
    "eval_metric": "aucpr",
    "early_stopping_rounds": 100,
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


@dataclass(frozen=True)
class BaselineResult:
    model: xgb.XGBClassifier
    features: list[str]
    best_iteration: int
    valid_prob: np.ndarray  # aligned with split.valid's rows
    valid_scores: Scores


def train_baseline(
    split: Split, device: str = "cpu", verbose: int | bool = 100, **overrides
) -> BaselineResult:
    """Fit on ``split.train``, early-stop and score on ``split.valid``.

    ``device="cuda"`` uses a GPU (Kaggle: Accelerator → GPU). ``overrides``
    replace entries of ``PARAMS``, e.g. ``n_estimators=50`` in tests.
    """
    features = feature_columns(split.train)
    model = xgb.XGBClassifier(**{**PARAMS, "device": device, **overrides})
    model.fit(
        split.train[features],
        split.train[TARGET],
        eval_set=[(split.valid[features], split.valid[TARGET])],
        verbose=verbose,
    )
    # With early stopping, predict_proba uses the best iteration's trees.
    valid_prob = model.predict_proba(split.valid[features])[:, 1]
    return BaselineResult(
        model=model,
        features=features,
        best_iteration=int(model.best_iteration),
        valid_prob=valid_prob,
        valid_scores=score(split.valid[TARGET].to_numpy(), valid_prob),
    )

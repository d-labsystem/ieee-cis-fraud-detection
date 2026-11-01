"""Final model: fixed tree count, seed ensemble (final step).

Once the validation month is folded into training there is nothing left to
early-stop on, so the number of trees comes from the validation runs instead.
Averaging a few seeds removes most of the seed-to-seed noise seen in Phases
4-7 (about +-0.003 PR-AUC at depth 12).
"""

from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from fraud.baseline import PARAMS, TARGET

# Validation best iterations at depth 12 were 2078 / 1614 / 1910 (mean ~1870)
# on 4 months of data; the final model trains on 5, so slightly more trees.
N_TREES = 2100
SEEDS: tuple[int, ...] = (0, 1, 2)


def train_fixed(
    train: pd.DataFrame, features: list[str], n_estimators: int, device: str = "cpu", seed: int = 0
) -> xgb.XGBClassifier:
    """PARAMS with exactly ``n_estimators`` trees and no early stopping."""
    model = xgb.XGBClassifier(**{
        **PARAMS, "n_estimators": n_estimators, "early_stopping_rounds": None,
        "device": device, "random_state": seed,
    })
    return model.fit(train[features], train[TARGET], verbose=False)


def train_ensemble(
    train: pd.DataFrame,
    features: list[str],
    n_estimators: int,
    device: str = "cpu",
    seeds: Sequence[int] = (0, 1, 2),
) -> list[xgb.XGBClassifier]:
    return [train_fixed(train, features, n_estimators, device, seed) for seed in seeds]


def ensemble_predict(models: list[xgb.XGBClassifier], df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Mean fraud probability over the models."""
    return np.mean([m.predict_proba(df[features])[:, 1] for m in models], axis=0)

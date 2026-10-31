"""Ranking metrics for an imbalanced label, reported with the prevalence.

A random classifier's PR-AUC equals the fraud rate, so raw PR-AUC partly
tracks the base rate. ``pr_lift`` (PR-AUC / prevalence) makes scores on
months with different fraud rates comparable.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import average_precision_score, roc_auc_score


class Scores(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n_transactions: int
    n_frauds: int
    prevalence: float
    roc_auc: float
    pr_auc: float
    pr_lift: float


def score(y_true: np.ndarray, y_prob: np.ndarray) -> Scores:
    y_true = np.asarray(y_true)
    n_frauds = int(y_true.sum())
    if n_frauds in (0, len(y_true)):
        raise ValueError("y_true must contain both frauds and legitimate transactions.")

    prevalence = n_frauds / len(y_true)
    # average_precision is the step-wise PR-AUC; it avoids the optimistic
    # linear interpolation of auc(recall, precision).
    pr_auc = float(average_precision_score(y_true, y_prob))
    return Scores(
        n_transactions=len(y_true),
        n_frauds=n_frauds,
        prevalence=prevalence,
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=pr_auc,
        pr_lift=pr_auc / prevalence,
    )

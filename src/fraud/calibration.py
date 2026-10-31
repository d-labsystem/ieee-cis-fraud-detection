"""Turn model scores into probabilities you can act on (Phase 5d).

A model can rank perfectly and still say "0.05" for transactions that are
fraud 35% of the time. Calibration fits a monotone map from score to observed
fraud rate:

- Platt scaling: logistic regression on the score's log-odds (2 parameters,
  strictly increasing, so ROC-AUC / PR-AUC are unchanged);
- isotonic regression: a free-form increasing step function (more flexible,
  but needs more data and creates ties, so the ranking can change slightly).

The calibrator must be fitted on data the model was not tuned on and then
judged on other data again; ``time_halves`` splits a month for that.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from fraud.metrics import score

# Predicted-probability bin edges. Fraud scores pile up near 0, so equal-width
# bins would put almost every transaction in the first one.
EDGES = (0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 0.9, 0.99, 1)
_EPS = 1e-12  # float64 keeps scores this small distinct, so no ties are created

Calibrator = Callable[[np.ndarray], np.ndarray]


def _logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def time_halves(dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks: rows before the median time (fit), and from it on (eval)."""
    first = np.asarray(dt) < np.median(dt)
    return first, ~first


def reliability_table(y: np.ndarray, prob: np.ndarray, edges=EDGES) -> pd.DataFrame:
    """Per probability bin: how many rows, mean prediction, actual fraud rate."""
    prob = np.asarray(prob, dtype=float)
    bins = pd.cut(prob, bins=np.asarray(edges), include_lowest=True)
    frame = pd.DataFrame({"bin": bins, "prob": prob, "isFraud": np.asarray(y)})
    out = frame.groupby("bin", observed=True).agg(
        transactions=("prob", "size"), mean_pred=("prob", "mean"), fraud_rate=("isFraud", "mean")
    )
    return out


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, edges=EDGES) -> float:
    """Row-weighted mean |mean prediction - fraud rate| over the bins."""
    table = reliability_table(y, prob, edges)
    weights = table["transactions"] / table["transactions"].sum()
    return float((weights * (table["mean_pred"] - table["fraud_rate"]).abs()).sum())


def calibration_scores(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Calibration (Brier, log loss, ECE) next to ranking (ROC-AUC, PR-AUC)."""
    ranking = score(y, prob)
    return {
        "brier": brier_score_loss(y, prob),
        "log_loss": log_loss(y, np.clip(prob, _EPS, 1 - _EPS)),
        "ece": expected_calibration_error(y, prob),
        "mean_pred": float(np.mean(prob)),
        "fraud_rate": float(np.mean(y)),
        "roc_auc": ranking.roc_auc,
        "pr_auc": ranking.pr_auc,
    }


@dataclass(frozen=True)
class Platt:
    """p_calibrated = sigmoid(slope * logit(p) + intercept)."""

    slope: float
    intercept: float

    def __call__(self, prob: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.slope * _logit(prob) + self.intercept)))


def fit_platt(y: np.ndarray, prob: np.ndarray) -> Platt:
    model = LogisticRegression(penalty=None).fit(_logit(prob).reshape(-1, 1), y)
    return Platt(slope=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]))


def fit_isotonic(y: np.ndarray, prob: np.ndarray) -> Calibrator:
    model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(prob, y)
    return model.predict


def expected_benefit(
    prob: np.ndarray, amount: np.ndarray, review_cost: float, false_alarm_cost: float
) -> np.ndarray:
    """$ expected from reviewing each transaction, under the 5c cost model.

    Reviewing saves the amount if it is fraud (probability p) and costs the
    review, plus the false-alarm cost if it is legitimate (probability 1 - p).
    Reviewing is worth it when this is positive; that needs calibrated p.
    """
    p = np.asarray(prob, dtype=float)
    return p * np.asarray(amount, dtype=float) - review_cost - (1 - p) * false_alarm_cost

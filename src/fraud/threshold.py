"""Turn scores into alerts: how many to review, and at what cost (Phase 5c).

A model only ranks transactions; the business picks how far down the ranking
analysts go. Two ways to choose:

- capacity: the team can review a fixed share of traffic, so the threshold is
  whatever score that share starts at (``capacity_table``);
- cost: each review costs money, each false alarm annoys a real customer and
  each missed fraud loses its amount, so pick the number of alerts with the
  lowest total cost (``cost_curve``).

Both are chosen on validation and applied unchanged to the test months.
"""

import numpy as np
import pandas as pd

SHARES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)


def _ranked(y: np.ndarray, prob: np.ndarray, amount: np.ndarray):
    """Labels, scores and amounts sorted from highest score down."""
    order = np.argsort(-np.asarray(prob), kind="stable")
    return np.asarray(y)[order].astype(bool), np.asarray(prob)[order], np.asarray(amount)[order]


def capacity_table(
    y: np.ndarray, prob: np.ndarray, amount: np.ndarray, shares=SHARES
) -> pd.DataFrame:
    """Recall/precision (by count and by $) if analysts review the top ``share``.

    ``frauds_per_100_extra`` is the precision of only the alerts added since
    the previous row: once it drops, extra reviewers mostly see legit traffic.
    """
    fraud, p, amt = _ranked(y, prob, amount)
    caught = np.cumsum(fraud)
    caught_amt = np.cumsum(np.where(fraud, amt, 0.0))
    rows, prev_k = [], 0
    for share in shares:
        k = max(1, int(round(share * len(p))))
        new = caught[k - 1] - (caught[prev_k - 1] if prev_k else 0)
        rows.append({
            "review_share": share,
            "alerts": k,
            "threshold": p[k - 1],
            "frauds_caught": int(caught[k - 1]),
            "recall": caught[k - 1] / fraud.sum(),
            "precision": caught[k - 1] / k,
            "amount_recall": caught_amt[k - 1] / amt[fraud].sum(),
            "frauds_per_100_extra": 100 * new / (k - prev_k) if k > prev_k else np.nan,
        })
        prev_k = k
    return pd.DataFrame(rows).set_index("review_share")


def cost_curve(
    y: np.ndarray,
    prob: np.ndarray,
    amount: np.ndarray,
    review_cost: float,
    false_alarm_cost: float,
) -> pd.DataFrame:
    """Total cost of alerting on the top k transactions, for every k from 0 to n.

    cost(k) = review_cost * k
            + false_alarm_cost * (legit transactions among the k alerts)
            + amount of every fraud not among the k alerts

    Assumes a reviewed fraud is always stopped and its full amount saved.
    """
    fraud, p, amt = _ranked(y, prob, amount)
    k = np.arange(len(p) + 1)
    caught_amt = np.concatenate([[0.0], np.cumsum(np.where(fraud, amt, 0.0))])
    false_alarms = np.concatenate([[0], np.cumsum(~fraud)])
    missed_amt = amt[fraud].sum() - caught_amt
    return pd.DataFrame({
        "alerts": k,
        "threshold": np.concatenate([[np.inf], p]),
        "review": review_cost * k,
        "false_alarm": false_alarm_cost * false_alarms,
        "missed_fraud": missed_amt,
        "total": review_cost * k + false_alarm_cost * false_alarms + missed_amt,
    })


def best_alerts(curve: pd.DataFrame) -> pd.Series:
    """The row of ``cost_curve`` with the lowest total cost."""
    return curve.loc[curve["total"].idxmin()]

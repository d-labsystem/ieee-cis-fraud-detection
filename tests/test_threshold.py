import numpy as np
import pandas as pd
import pytest

from fraud.threshold import best_alerts, capacity_table, cost_curve

# Scores already in descending order to keep the expected values readable.
Y = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0])
PROB = np.array([0.95, 0.90, 0.80, 0.70, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05])
AMT = np.array([100.0, 50.0, 10.0, 200.0, 10.0, 10.0, 10.0, 650.0, 10.0, 10.0])


def test_capacity_table():
    out = capacity_table(Y, PROB, AMT, shares=(0.2, 0.4, 1.0))
    assert out["alerts"].tolist() == [2, 4, 10]
    assert out["frauds_caught"].tolist() == [2, 3, 4]
    assert out["recall"].tolist() == pytest.approx([0.5, 0.75, 1.0])
    assert out["precision"].tolist() == pytest.approx([1.0, 0.75, 0.4])
    assert out["amount_recall"].tolist() == pytest.approx([0.15, 0.35, 1.0])
    # Alerts 3-4 hold one fraud out of two; alerts 5-10 one out of six.
    assert out["frauds_per_100_extra"].tolist() == pytest.approx([100, 50, 100 / 6])
    assert out.loc[0.4, "threshold"] == pytest.approx(0.70)


def test_capacity_table_any_row_order():
    rng = np.random.default_rng(0).permutation(len(Y))
    pd.testing.assert_frame_equal(
        capacity_table(Y[rng], PROB[rng], AMT[rng], shares=(0.2, 0.4)),
        capacity_table(Y, PROB, AMT, shares=(0.2, 0.4)),
    )


def test_cost_curve_endpoints():
    curve = cost_curve(Y, PROB, AMT, review_cost=5, false_alarm_cost=2)
    assert len(curve) == len(Y) + 1
    # No alerts: every fraud is lost.
    assert curve.loc[0, "total"] == pytest.approx(1000)
    # Alert on everything: 10 reviews, 6 false alarms, nothing lost.
    assert curve.loc[10, "total"] == pytest.approx(10 * 5 + 6 * 2)


def test_best_alerts_goes_deep_for_a_large_fraud():
    # The $650 fraud at rank 8 is worth reviewing the 3 legit rows before it.
    best = best_alerts(cost_curve(Y, PROB, AMT, review_cost=5, false_alarm_cost=2))
    assert best["alerts"] == 8
    assert best["total"] == pytest.approx(8 * 5 + 4 * 2)


def test_best_alerts_with_expensive_reviews():
    # At $300 per review, only the $200+ frauds would pay off: alerting on
    # nobody costs 1000, and no k does better.
    best = best_alerts(cost_curve(Y, PROB, AMT, review_cost=300, false_alarm_cost=0))
    assert best["alerts"] == 0

import numpy as np
import pytest

from fraud.calibration import (
    calibration_scores,
    expected_benefit,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    reliability_table,
    time_halves,
)


@pytest.fixture(scope="module")
def overconfident():
    """True fraud probability p; the 'model' reports sigmoid(2 * logit(p))."""
    rng = np.random.default_rng(0)
    true_logit = rng.normal(-3.5, 1.5, size=50_000)
    y = (rng.random(true_logit.size) < 1 / (1 + np.exp(-true_logit))).astype(int)
    raw = 1 / (1 + np.exp(-(2 * true_logit)))
    return y, raw


def test_time_halves():
    first, second = time_halves(np.array([5, 1, 4, 2, 3, 6]))
    assert first.tolist() == [False, True, False, True, True, False]  # median 3.5
    assert (first ^ second).all()


def test_reliability_table_bins():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.0005, 0.002, 0.95, 0.995])
    table = reliability_table(y, prob)
    assert table["transactions"].tolist() == [1, 1, 1, 1]
    assert table["fraud_rate"].tolist() == [0, 0, 1, 1]


def test_ece_zero_when_predictions_match_rates():
    y = np.array([0] * 9 + [1])
    assert expected_calibration_error(y, np.full(10, 0.1)) == pytest.approx(0)


def test_platt_recovers_the_distortion(overconfident):
    y, raw = overconfident
    platt = fit_platt(y, raw)
    assert platt.slope == pytest.approx(0.5, abs=0.05)
    assert platt.intercept == pytest.approx(0, abs=0.2)


def test_calibration_improves_scores_but_not_ranking(overconfident):
    y, raw = overconfident
    before = calibration_scores(y, raw)
    for calibrate in (fit_platt(y, raw), fit_isotonic(y, raw)):
        after = calibration_scores(y, calibrate(raw))
        assert after["log_loss"] < before["log_loss"]
        assert after["ece"] < before["ece"]
        assert after["roc_auc"] == pytest.approx(before["roc_auc"], abs=0.002)
    # Platt is strictly increasing: the ranking is exactly the same.
    assert calibration_scores(y, fit_platt(y, raw)(raw))["roc_auc"] == pytest.approx(before["roc_auc"])


def test_isotonic_stays_in_unit_interval(overconfident):
    y, raw = overconfident
    out = fit_isotonic(y, raw)(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]))
    assert ((out >= 0) & (out <= 1)).all()


def test_expected_benefit():
    # 10% chance of a $500 fraud: 50 - 3 - 0.9 * 10 = 38 -> review.
    # 10% chance of a $20 fraud:   2 - 3 - 9 = -10     -> let it through.
    out = expected_benefit(np.array([0.1, 0.1]), np.array([500.0, 20.0]), 3, 10)
    assert out.tolist() == pytest.approx([38, -10])

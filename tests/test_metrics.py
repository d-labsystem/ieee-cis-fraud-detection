import numpy as np
import pytest

from fraud.metrics import score


def test_perfect_ranking():
    s = score(np.array([0, 0, 0, 1]), np.array([0.1, 0.2, 0.3, 0.9]))
    assert s.roc_auc == pytest.approx(1.0)
    assert s.pr_auc == pytest.approx(1.0)
    assert s.prevalence == pytest.approx(0.25)
    assert s.pr_lift == pytest.approx(4.0)
    assert (s.n_transactions, s.n_frauds) == (4, 1)


def test_reversed_ranking():
    s = score(np.array([0, 0, 0, 1]), np.array([0.9, 0.8, 0.7, 0.1]))
    assert s.roc_auc == pytest.approx(0.0)
    # The only fraud is ranked last: precision at full recall = prevalence.
    assert s.pr_auc == pytest.approx(0.25)
    assert s.pr_lift == pytest.approx(1.0)


def test_constant_scores_give_lift_one():
    y = np.array([0] * 90 + [1] * 10)
    s = score(y, np.full(100, 0.5))
    assert s.roc_auc == pytest.approx(0.5)
    assert s.pr_lift == pytest.approx(1.0)


@pytest.mark.parametrize("y", [np.zeros(5), np.ones(5)])
def test_single_class_raises(y):
    with pytest.raises(ValueError, match="both"):
        score(y, np.linspace(0, 1, 5))

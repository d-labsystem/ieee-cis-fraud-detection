import numpy as np
import pandas as pd
import pytest

from fraud.drift import (
    adversarial_validation,
    client_segments,
    psi,
    psi_table,
    segment_week_scores,
    week_scores,
)


@pytest.fixture(scope="module")
def periods():
    """`stable` has the same distribution in both periods; `shifted` moves by 1 sd."""
    rng = np.random.default_rng(0)
    n = 4_000
    ref = pd.DataFrame({"stable": rng.normal(size=n), "shifted": rng.normal(size=n),
                        "browser": pd.Categorical(rng.choice(["chrome", "safari"], size=n))})
    cur = pd.DataFrame({"stable": rng.normal(size=n), "shifted": rng.normal(1, 1, size=n),
                        "browser": pd.Categorical(rng.choice(["chrome", "safari"], size=n))})
    return ref, cur


def test_psi_same_distribution_is_small(periods):
    ref, cur = periods
    assert psi(ref["stable"], cur["stable"]) < 0.02
    assert psi(ref["browser"], cur["browser"]) < 0.02


def test_psi_shift_is_large(periods):
    ref, cur = periods
    assert psi(ref["shifted"], cur["shifted"]) > 0.5


def test_psi_identical_is_zero():
    s = pd.Series([1.0, 2.0, 3.0, np.nan] * 25)
    assert psi(s, s) == pytest.approx(0)


def test_psi_counts_missing_and_unseen():
    # Half the current rows turn missing: a big shift even though values match.
    ref = pd.Series(np.arange(100, dtype=float))
    cur = pd.Series(np.r_[np.arange(50, dtype=float), [np.nan] * 50])
    assert psi(ref, cur) > 1
    # A category never seen in the reference period counts as drift.
    old = pd.Series(["chrome 65"] * 100)
    new = pd.Series(["chrome 65"] * 50 + ["chrome 66"] * 50)
    assert psi(old, new) > 1


def test_psi_table_sorted(periods):
    ref, cur = periods
    table = psi_table(ref, cur, ["stable", "shifted", "browser"])
    assert table.index[0] == "shifted"


def test_adversarial_validation_finds_the_shift(periods):
    ref, cur = periods
    auc, gain = adversarial_validation(ref, cur, ["stable", "shifted", "browser"])
    assert 0.65 < auc < 0.85  # a 1-sd shift separates the periods only partly
    assert gain.index[0] == "shifted"
    assert gain.sum() == pytest.approx(100)


def test_adversarial_validation_no_shift():
    rng = np.random.default_rng(1)
    ref, cur = (pd.DataFrame({"x": rng.normal(size=3_000)}) for _ in range(2))
    auc, _ = adversarial_validation(ref, cur, ["x"])
    assert abs(auc - 0.5) < 0.05


def test_week_scores():
    day = np.repeat(np.arange(121, 135), 10)  # two weeks
    y = np.tile([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 14)
    prob = np.where(y == 1, 0.9, 0.1)
    out = week_scores(y, prob, day)
    assert out.index.tolist() == [0, 1]
    assert out["first_day"].tolist() == [121, 128]
    assert out["roc_auc"].tolist() == pytest.approx([1, 1])


def test_segment_week_scores_align_weeks():
    # "late" only starts on day 130, i.e. in week 1 of the full period.
    day = np.r_[np.repeat(np.arange(121, 135), 10), np.repeat(np.arange(130, 135), 10)]
    y = np.r_[np.tile([1] + [0] * 9, 14), np.tile([1] + [0] * 9, 5)]
    prob = np.where(y == 1, 0.9, 0.1)
    segment = np.r_[["early"] * 140, ["late"] * 50]
    out = segment_week_scores(y, prob, day, segment)
    assert out.index.tolist() == [(0, "early"), (1, "early"), (1, "late")]
    # Week 1 (days 128-134): 70 early rows + 50 late rows.
    assert out.loc[(1, "late"), "share_of_week"] == pytest.approx(50 / 120)
    assert out.loc[(0, "early"), "share_of_week"] == pytest.approx(1)


def test_week_scores_keeps_single_class_weeks():
    day = np.repeat([121, 128], 5)
    y = np.r_[[1, 0, 0, 0, 0], [0] * 5]  # no fraud in week 1
    out = week_scores(y, np.linspace(0, 1, 10), day)
    assert out["transactions"].tolist() == [5, 5]
    assert np.isnan(out.loc[1, "roc_auc"])


def test_client_segments_use_known_months_only():
    df = pd.DataFrame({
        "card1": [1, 2, 3, 1, 2, 3, 4, 3],
        "addr1": [10.0] * 8,
        "D1n": [5.0] * 8,
        "month": [0, 0, 0, 4, 4, 4, 4, 5],
        "isFraud": [0, 1, 0, 0, 0, 1, 1, 1],
    })
    seg = client_segments(df, known_months=(0,), target_months=(4,))
    assert seg.index.tolist() == [3, 4, 5, 6]
    assert seg.tolist() == ["known, no fraud", "known, fraud", "known, no fraud", "new client"]

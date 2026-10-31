import numpy as np
import pandas as pd
import pytest

from fraud.data.load import SECONDS_PER_DAY
from fraud.features.entity import ENTITIES, add_entity_features, add_entity_rank_features
from fraud.features.windows import window_counts

DAY = SECONDS_PER_DAY
ENTITY_COLS = ["card1", "addr1", "P_emaildomain", "R_emaildomain",
               "DeviceInfo", "id_30", "id_31", "id_33"]


def _df(dt, amt, **cols) -> pd.DataFrame:
    """Rows at times ``dt``; entity columns default to missing."""
    n = len(dt)
    data = {c: cols.get(c, [np.nan] * n) for c in ENTITY_COLS}
    return pd.DataFrame({"TransactionDT": dt, "TransactionAmt": amt, **data})


# ---------- window_counts ----------


def test_window_counts_basic():
    codes = np.array([0, 0, 0, 1, 0])
    dt = np.array([0, 100, 200, 150, 400])
    # Window of 150s: code 0 at t=200 sees t=100 (and t=50 would count too).
    assert window_counts(codes, dt, 150).tolist() == [0, 1, 1, 0, 0]
    assert window_counts(codes, dt, 1_000).tolist() == [0, 1, 2, 0, 3]


def test_window_counts_any_input_order():
    codes = np.array([0, 1, 0, 0])
    dt = np.array([300, 0, 0, 100])
    assert window_counts(codes, dt, 1_000).tolist() == [2, 0, 0, 1]


def test_window_counts_start_is_inclusive():
    assert window_counts(np.zeros(2, int), np.array([0, 60]), 60).tolist() == [0, 1]


def test_window_counts_rejects_huge_dt():
    with pytest.raises(ValueError, match="below"):
        window_counts(np.zeros(1, int), np.array([10**9]), 60)


# ---------- add_entity_features ----------


def test_adds_two_features_per_entity():
    out = add_entity_features(_df([DAY], [10.0]))
    for prefix in ENTITIES:
        assert f"{prefix}_n_24h" in out and f"{prefix}_share_30d" in out


def test_card1_velocity_and_share():
    df = _df(
        dt=[DAY, DAY + 10, DAY + 20, 3 * DAY],
        amt=[1.0, 2.0, 3.0, 4.0],
        card1=[7, 7, 8, 7],
    )
    out = add_entity_features(df)
    assert out["card1_n_24h"].tolist() == [0, 1, 0, 0]  # last row: >24h after the others
    # Shares: earlier same-card rows / all earlier rows, within 30 days.
    np.testing.assert_allclose(out["card1_share_30d"], [np.nan, 1.0, 0.0, 2 / 3])


def test_exact_amount_groups_by_cents_and_third_decimal():
    df = _df(dt=[DAY, DAY + 1, DAY + 2, DAY + 3], amt=[49.99, 49.99, 49.995, 50.0])
    out = add_entity_features(df)
    assert out["amt_n_24h"].tolist() == [0, 1, 0, 0]


def test_missing_entity_gives_nan_not_a_group():
    df = _df(dt=[DAY, DAY + 1], amt=[1.0, 2.0], P_emaildomain=[None, None])
    out = add_entity_features(df)
    assert out["P_email_n_24h"].isna().all()
    assert out["P_email_share_30d"].isna().all()


def test_device_fingerprint_combines_columns():
    df = _df(
        dt=[DAY, DAY + 1, DAY + 2],
        amt=[1.0, 2.0, 3.0],
        DeviceInfo=["SM-G930V", "SM-G930V", "SM-G930V"],
        id_31=["chrome", "chrome", "safari"],
    )
    out = add_entity_features(df)
    # Same model, different browser -> different fingerprint.
    assert out["device_n_24h"].tolist() == [0, 1, 0]


def test_future_rows_do_not_change_past_features():
    df = _df(dt=[DAY, DAY + 10], amt=[5.0, 5.0], card1=[1, 1])
    before = add_entity_features(df)
    future = _df(dt=[DAY + 20], amt=[5.0], card1=[1])
    after = add_entity_features(pd.concat([df, future], ignore_index=True))
    pd.testing.assert_frame_equal(after.iloc[:2], before)


# ---------- entity rank (Phase 7e) ----------


def _rank_df(days, emails) -> pd.DataFrame:
    df = _df([d * DAY for d in days], [10.0] * len(days), P_emaildomain=emails)
    return add_entity_features(df.assign(day=days))


def test_rank_replaces_share_and_is_causal():
    days = [1, 1, 1, 2, 2, 2, 3]
    emails = ["gmail", "gmail", "yahoo", "yahoo", "yahoo", "gmail", "yahoo"]
    out = add_entity_rank_features(_rank_df(days, emails))
    assert "P_email_rank_30d" in out and "P_email_share_30d" not in out
    rank = out["P_email_rank_30d"].tolist()
    # Day 1: nothing earlier -> NaN. Day 2 sees day 1 (gmail 2, yahoo 1).
    assert np.isnan(rank[:3]).all()
    assert rank[3:6] == [2, 2, 1]
    # Day 3 sees days 1-2: yahoo 3, gmail 3 -> tie, both rank 1.
    assert rank[6] == 1


def test_rank_ignores_missing_and_ranks_unseen_last():
    days = [1, 1, 1, 1, 2, 2]
    emails = [np.nan, np.nan, np.nan, "gmail", "aol", np.nan]
    rank = add_entity_rank_features(_rank_df(days, emails))["P_email_rank_30d"].tolist()
    assert rank[4] == 2   # aol unseen in the window: after gmail, not after the 3 missing rows
    assert np.isnan(rank[5])


def test_rank_window_is_30_days():
    days = [1, 40, 40]
    emails = ["aol", "gmail", "aol"]
    # On day 40 the window is days 10-39: day 1's aol row has dropped out.
    rank = add_entity_rank_features(_rank_df(days, emails))["P_email_rank_30d"].tolist()
    assert np.isnan(rank[1:]).all()


def test_rank_is_stable_when_volume_grows():
    """The point of 7e: yahoo's share moves from 60% to 70%, its rank stays 1."""
    days = list(np.repeat(np.arange(1, 21), 10))
    emails = (["gmail"] * 4 + ["yahoo"] * 6) * 10 + (["gmail"] * 3 + ["yahoo"] * 7) * 10
    df = _rank_df(days, emails)
    out = add_entity_rank_features(df)
    yahoo = (out["P_emaildomain"] == "yahoo") & (out["day"] >= 2)
    assert out.loc[yahoo, "P_email_rank_30d"].eq(1).all()
    share = df.loc[yahoo, "P_email_share_30d"]
    assert share.iloc[-1] - share.iloc[0] > 0.05

import numpy as np
import pandas as pd
import pytest

from fraud.data.load import SECONDS_PER_DAY
from fraud.features.client import add_client_features, client_id

HOUR = 3_600


def _df(rows) -> pd.DataFrame:
    """rows: (card1, addr1, D1, TransactionDT, TransactionAmt, P_emaildomain)."""
    df = pd.DataFrame(
        rows, columns=["card1", "addr1", "D1", "TransactionDT", "TransactionAmt", "P_emaildomain"]
    )
    df["day"] = df["TransactionDT"] // SECONDS_PER_DAY
    return df


@pytest.fixture
def one_client() -> pd.DataFrame:
    # Card first seen on day 0: D1 grows with the day, so D1n stays 0.
    t0 = 10 * SECONDS_PER_DAY
    return _df(
        [
            (1, 100.0, 10.0, t0, 10.0, "a.com"),
            (1, 100.0, 10.0, t0 + HOUR // 2, 20.0, "a.com"),
            (1, 100.0, 10.0, t0 + HOUR, 60.0, "b.com"),
            (1, 100.0, 12.0, t0 + 2 * SECONDS_PER_DAY, 30.0, "a.com"),
        ]
    )


def test_d1n_is_constant_for_one_client(one_client):
    out = add_client_features(one_client)
    assert out["D1n"].tolist() == [0, 0, 0, 0]
    assert len(set(client_id(out))) == 1


def test_expanding_amount_stats(one_client):
    out = add_client_features(one_client)
    assert out["uid_n_prev"].tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(out["uid_amt_mean_prev"], [np.nan, 10, 15, 30])
    np.testing.assert_allclose(out["uid_amt_std_prev"], [np.nan, np.nan, np.sqrt(50), np.sqrt(700)])
    np.testing.assert_allclose(out["uid_amt_ratio"], [np.nan, 2, 4, 1])


def test_days_since_prev(one_client):
    out = add_client_features(one_client)
    np.testing.assert_allclose(
        out["uid_days_since_prev"], [np.nan, 1 / 48, 1 / 48, 2 - 1 / 24], rtol=1e-6
    )


def test_velocity_counts(one_client):
    out = add_client_features(one_client)
    # Row 3 (t0 + 1h): rows at t0 and t0 + 30min are within the past hour.
    assert out["uid_n_1h"].tolist() == [0, 1, 2, 0]
    assert out["uid_n_24h"].tolist() == [0, 1, 2, 0]
    assert out["uid_n_7d"].tolist() == [0, 1, 2, 3]


def test_new_email(one_client):
    out = add_client_features(one_client)
    assert out["uid_new_email"].tolist() == [0, 0, 1, 0]
    assert out["uid_n_emails_prev"].tolist() == [0, 1, 1, 2]


def test_other_clients_do_not_count(one_client):
    other = _df([(2, 100.0, 10.0, 10 * SECONDS_PER_DAY + 100, 999.0, "a.com")])
    out = add_client_features(pd.concat([one_client, other], ignore_index=True))
    assert out["uid_n_prev"].tolist() == [0, 1, 2, 3, 0]
    assert out["uid_n_1h"].iloc[1] == 1


def test_missing_addr1_still_groups():
    t = 10 * SECONDS_PER_DAY
    df = _df([(1, np.nan, 10.0, t, 5.0, None), (1, np.nan, 10.0, t + 60, 5.0, None)])
    out = add_client_features(df)
    assert out["uid_n_prev"].tolist() == [0, 1]
    assert out["uid_new_email"].tolist() == [0, 0]  # missing email == missing email


def test_row_order_does_not_matter(one_client):
    expected = add_client_features(one_client)
    shuffled = one_client.sample(frac=1, random_state=0)
    out = add_client_features(shuffled).loc[expected.index]
    pd.testing.assert_frame_equal(out, expected[out.columns])


def test_future_rows_do_not_change_past_features(one_client):
    """Causality: appending later transactions leaves earlier rows unchanged."""
    before = add_client_features(one_client)
    future = _df([(1, 100.0, 15.0, 15 * SECONDS_PER_DAY, 5000.0, "z.com")])
    after = add_client_features(pd.concat([one_client, future], ignore_index=True))
    pd.testing.assert_frame_equal(after.iloc[:4], before)


def test_duplicate_index_raises(one_client):
    with pytest.raises(ValueError, match="unique"):
        add_client_features(pd.concat([one_client, one_client]))

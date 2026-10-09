import pandas as pd
import pytest

from fraud.data.sample import IDENTITY_FILE, TRANSACTION_FILE, make_sample


@pytest.fixture
def raw_dir(tmp_path):
    """Write a tiny fake IEEE-CIS raw folder: 5 transactions, 4 identity rows."""
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "isFraud": [0, 1, 0, 0, 1],
            "TransactionDT": [86_400, 90_000, 172_800, 200_000, 300_000],
            "TransactionAmt": [10.0, 25.5, 99.99, 5.0, 42.0],
        }
    ).to_csv(raw / TRANSACTION_FILE, index=False)
    # IDs 2 and 4 fall inside a 4-row sample; 5 and 9 fall outside it.
    pd.DataFrame(
        {
            "TransactionID": [2, 4, 5, 9],
            "DeviceType": ["mobile", "desktop", "mobile", "desktop"],
        }
    ).to_csv(raw / IDENTITY_FILE, index=False)
    return raw


def test_keeps_first_n_transactions(raw_dir, tmp_path):
    out = tmp_path / "out"
    make_sample(raw_dir, out, n_transactions=4)
    transactions = pd.read_parquet(out / "transactions.parquet")
    assert transactions["TransactionID"].tolist() == [1, 2, 3, 4]


def test_identity_selected_by_key_not_position(raw_dir, tmp_path):
    out = tmp_path / "out"
    make_sample(raw_dir, out, n_transactions=4)
    identity = pd.read_parquet(out / "identity.parquet")
    # Positionally, the first 4 identity rows would be IDs 2, 4, 5, 9.
    assert sorted(identity["TransactionID"].tolist()) == [2, 4]


def test_summary_numbers(raw_dir, tmp_path):
    summary = make_sample(raw_dir, tmp_path / "out", n_transactions=4)
    assert summary.n_transactions == 4
    assert summary.n_identity == 2
    assert summary.identity_coverage == pytest.approx(0.5)
    assert summary.fraud_rate == pytest.approx(0.25)
    assert summary.first_day == pytest.approx(1.0)


def test_rerun_overwrites_cleanly(raw_dir, tmp_path):
    out = tmp_path / "out"
    make_sample(raw_dir, out, n_transactions=4)
    summary = make_sample(raw_dir, out, n_transactions=2)
    assert summary.n_transactions == 2
    assert len(pd.read_parquet(out / "transactions.parquet")) == 2


def test_rejects_unsorted_file(raw_dir, tmp_path):
    path = raw_dir / TRANSACTION_FILE
    df = pd.read_csv(path)
    df.loc[0, "TransactionDT"] = 999_999
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not sorted"):
        make_sample(raw_dir, tmp_path / "out", n_transactions=4)


@pytest.mark.parametrize("bad_n", [0, -1])
def test_rejects_non_positive_n(raw_dir, tmp_path, bad_n):
    with pytest.raises(ValueError, match="must be positive"):
        make_sample(raw_dir, tmp_path / "out", n_transactions=bad_n)
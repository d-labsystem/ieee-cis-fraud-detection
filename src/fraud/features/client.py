"""Client-level features (Phase 4a).

IEEE-CIS labels clients, not single transactions: once a card is reported,
its later transactions are labelled fraud too. There is no client column, so
we approximate one:

    uid = card1 + addr1 + D1n,   D1n = day - D1

D1 is "days since the card was first seen", so ``day - D1`` is the day the
card was first seen. It stays constant across one client's transactions.

Every feature is causal: a row only uses that client's *earlier* rows, so
the same code is valid at transaction time and leaks nothing from the future.
"""

import numpy as np
import pandas as pd

from fraud.data.load import SECONDS_PER_DAY
from fraud.features.windows import window_counts

KEY_COLS: tuple[str, ...] = ("card1", "addr1", "D1n")

# Look-back windows for velocity counts, in seconds.
WINDOWS: dict[str, int] = {"1h": 3_600, "24h": SECONDS_PER_DAY, "7d": 7 * SECONDS_PER_DAY}


def client_id(df: pd.DataFrame) -> np.ndarray:
    """Integer client id per row. Missing key parts (e.g. addr1, absent for
    most of product C) form their own group instead of dropping the row."""
    return df.groupby(list(KEY_COLS), dropna=False, sort=False).ngroup().to_numpy()


def add_client_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``D1n`` and the ``uid_*`` features added."""
    if not df.index.is_unique:
        raise ValueError("df index must be unique to align features back to rows.")

    out = df.copy(deep=False)
    out["D1n"] = (out["day"] - out["D1"]).astype(np.float32)

    uid = client_id(out)
    dt = out["TransactionDT"].to_numpy(dtype=np.int64)

    # Work in (client, time) order; `inverse` maps results back to df's order.
    order = np.lexsort((dt, uid))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))

    s = pd.DataFrame(
        {
            "uid": uid[order],
            "dt": dt[order],
            "amt": out["TransactionAmt"].to_numpy(dtype=np.float64)[order],
            "email": out["P_emaildomain"].to_numpy()[order],
        }
    )
    by_uid = s.groupby("uid", sort=False)

    # Expanding stats over earlier rows only: cumulative sums minus the current row.
    n_prev = by_uid.cumcount().to_numpy()
    sum_prev = by_uid["amt"].cumsum().to_numpy() - s["amt"].to_numpy()
    sq = s["amt"].to_numpy() ** 2
    sq_prev = pd.Series(sq).groupby(s["uid"]).cumsum().to_numpy() - sq
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_prev = np.where(n_prev > 0, sum_prev / n_prev, np.nan)
        # Sample variance (ddof=1) from sums: (Σx² - n·mean²) / (n - 1).
        var_prev = np.where(n_prev > 1, (sq_prev - n_prev * mean_prev**2) / (n_prev - 1), np.nan)
        amt_ratio = s["amt"].to_numpy() / mean_prev

    # An email domain this client has not used before. NaN counts as a value.
    first_email = ~s.duplicated(["uid", "email"]).to_numpy()
    emails_prev = pd.Series(first_email).groupby(s["uid"]).cumsum().to_numpy() - first_email

    features = {
        "uid_n_prev": n_prev,
        "uid_amt_mean_prev": mean_prev,
        "uid_amt_std_prev": np.sqrt(np.clip(var_prev, 0, None)),
        "uid_amt_ratio": amt_ratio,
        "uid_days_since_prev": by_uid["dt"].diff().to_numpy() / SECONDS_PER_DAY,
        "uid_n_emails_prev": emails_prev,
        "uid_new_email": first_email & (n_prev > 0),
    }
    for name, values in features.items():
        out[name] = np.asarray(values, dtype=np.float32)[inverse]
    # Velocity: window_counts works in input order, so no `inverse` needed.
    for name, seconds in WINDOWS.items():
        out[f"uid_n_{name}"] = window_counts(uid, dt, seconds).astype(np.float32)
    return out

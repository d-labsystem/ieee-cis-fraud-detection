"""Entity frequency features (Phase 4b).

Client history is empty on a client's first transaction. These features
borrow history from the other entities in the transaction instead: how busy
this card BIN, region, email domain, device or exact amount has been.

Two numbers per entity, both causal (earlier rows only):

- ``<entity>_n_24h``: earlier transactions with the same value in the last
  24 hours. Velocity; a burst on one device is a card-testing sign.
- ``<entity>_share_30d``: those in the last 30 days as a share of *all*
  transactions in the same 30 days. A frequency encoding that stays
  comparable when total volume changes (the holiday month has ~50% more
  transactions, which would inflate a raw count) and at the start of the data,
  where both numerator and denominator are cut short.
"""

import numpy as np
import pandas as pd

from fraud.data.load import SECONDS_PER_DAY
from fraud.features.windows import window_counts

# Feature-name prefix -> columns whose combined value identifies the entity.
ENTITIES: dict[str, tuple[str, ...]] = {
    "card1": ("card1",),  # card issuer / BIN-like group, shared by many clients
    "addr1": ("addr1",),  # billing region
    "P_email": ("P_emaildomain",),
    "R_email": ("R_emaildomain",),
    # No device id exists; model + OS + browser + screen size approximates one.
    "device": ("DeviceInfo", "id_30", "id_31", "id_33"),
    "amt": ("_amt_milli",),  # exact amount; popular prices recur
}

VELOCITY_SECONDS = SECONDS_PER_DAY
SHARE_SECONDS = 30 * SECONDS_PER_DAY


def _entity_codes(df: pd.DataFrame, cols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Integer code per row, and a mask of rows where every column is missing."""
    codes = df.groupby(list(cols), dropna=False, sort=False, observed=True).ngroup().to_numpy()
    missing = df[list(cols)].isna().all(axis=1).to_numpy()
    return codes, missing


def add_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``<entity>_n_24h`` and ``<entity>_share_30d`` added."""
    out = df.copy(deep=False)
    dt = out["TransactionDT"].to_numpy(dtype=np.int64)

    # Group amounts by exact value. float32 can't hold a 3rd decimal above
    # ~$16K, but amounts that large are rare and still group consistently.
    keys = out.assign(
        _amt_milli=(out["TransactionAmt"].astype(np.float64) * 1000).round().astype(np.int64)
    )

    all_rows = np.zeros(len(out), dtype=np.int64)
    total_30d = window_counts(all_rows, dt, SHARE_SECONDS).astype(np.float64)

    for prefix, cols in ENTITIES.items():
        codes, missing = _entity_codes(keys, cols)
        n_24h = window_counts(codes, dt, VELOCITY_SECONDS).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = window_counts(codes, dt, SHARE_SECONDS) / total_30d
        # A missing value is not an entity: leave NaN and let XGBoost route it.
        n_24h[missing] = np.nan
        share[missing | (total_30d == 0)] = np.nan
        out[f"{prefix}_n_24h"] = n_24h
        out[f"{prefix}_share_30d"] = share.astype(np.float32)
    return out

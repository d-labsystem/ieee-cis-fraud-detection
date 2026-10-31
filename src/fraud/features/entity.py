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
from scipy.stats import rankdata

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


RANK_DAYS = 30


def _daily_rank(day: np.ndarray, codes: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Per row: popularity rank of its entity over the previous ``RANK_DAYS`` days.

    1 = the most common entity in that window; ties share the best rank; an
    entity not seen in the window ranks after every seen one. The current day
    is excluded, so the rank is causal at day granularity. Missing values are
    not an entity and are not counted; rows on a day with an empty window
    (the first day of data) get NaN.
    """
    d = day - day.min()
    counts = np.zeros((d.max() + 1, codes.max() + 1), dtype=np.int32)
    np.add.at(counts, (d[~missing], codes[~missing]), 1)
    cum = np.vstack([np.zeros((1, counts.shape[1]), dtype=np.int64), np.cumsum(counts, axis=0)])
    days = np.arange(counts.shape[0])
    window = cum[days] - cum[np.maximum(days - RANK_DAYS, 0)]  # days [d - 30, d - 1]
    ranks = rankdata(-window, axis=1, method="min").astype(np.float32)
    ranks[window.sum(axis=1) == 0] = np.nan
    return ranks[d, codes]


def add_entity_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace each ``<entity>_share_30d`` with ``<entity>_rank_30d`` (Phase 7e).

    A share drifts with volume: gmail is 38% of traffic in one month and 41%
    in the next, so the exact share also tells the model *when* a row is from
    (PSI 3.4, a top separator in adversarial validation). The rank (gmail is
    #1 either way) keeps "how common is this entity" without the clock.
    """
    out = df.copy(deep=False)
    keys = out.assign(
        _amt_milli=(out["TransactionAmt"].astype(np.float64) * 1000).round().astype(np.int64)
    )
    day = out["day"].to_numpy(dtype=np.int64)
    for prefix, cols in ENTITIES.items():
        codes, missing = _entity_codes(keys, cols)
        rank = _daily_rank(day, codes, missing)
        rank[missing] = np.nan
        out[f"{prefix}_rank_30d"] = rank
        out = out.drop(columns=f"{prefix}_share_30d")
    return out

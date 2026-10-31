"""Drop near-duplicate V columns (Phase 4d).

The 339 V columns fall into 14 blocks that are missing on exactly the same
rows (one engineered source each). Inside a block many columns are close
copies of each other. Keeping one per near-duplicate group gives fewer
features to train, explain and monitor for drift.

Blocks and correlations are measured on the training months only, so the
choice of columns uses no validation or test data.
"""

import numpy as np
import pandas as pd

from fraud.split import TRAIN_MONTHS

V_PATTERN = r"^V\d+$"
# Measured on train months: 0.98 keeps 322 of 339, 0.95 keeps 279, 0.90 keeps 228.
THRESHOLD = 0.90


def v_blocks(fit: pd.DataFrame) -> list[list[str]]:
    """V columns grouped by identical missing-value mask, in column order."""
    cols = list(fit.columns[fit.columns.str.match(V_PATTERN)])
    masks = fit[cols].isna().to_numpy()
    keys = [np.packbits(masks[:, j]).tobytes() for j in range(len(cols))]
    groups: dict[bytes, list[str]] = {}
    for col, key in zip(cols, keys):
        groups.setdefault(key, []).append(col)
    return list(groups.values())


def select_v_columns(fit: pd.DataFrame, threshold: float = THRESHOLD) -> list[str]:
    """Greedy, per block: keep a column unless |corr| >= threshold with one
    already kept. Constant columns are dropped (they can't split anything)."""
    kept: list[str] = []
    for cols in v_blocks(fit):
        x = fit[cols].dropna().to_numpy(dtype=np.float64)  # same rows for the whole block
        if len(x) < 2:
            continue
        sd = x.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.abs(np.corrcoef(x, rowvar=False)).reshape(len(cols), len(cols))
        block_kept: list[int] = []
        for j in range(len(cols)):
            if sd[j] > 0 and all(corr[j, i] < threshold for i in block_kept):
                block_kept.append(j)
        kept += [cols[j] for j in block_kept]
    return kept


def drop_correlated_v(
    df: pd.DataFrame,
    threshold: float = THRESHOLD,
    fit_months: tuple[int, ...] = TRAIN_MONTHS,
) -> pd.DataFrame:
    """Return ``df`` without the V columns that ``select_v_columns`` drops,
    choosing them from the rows in ``fit_months`` only."""
    fit = df[df["month"].isin(fit_months)]
    if fit.empty:
        raise ValueError(f"No rows in fit months {fit_months}.")
    keep = set(select_v_columns(fit, threshold))
    drop = [c for c in df.columns[df.columns.str.match(V_PATTERN)] if c not in keep]
    return df.drop(columns=drop)

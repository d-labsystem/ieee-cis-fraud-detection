"""Causal sliding-window counts, shared by the feature steps."""

import numpy as np

# code * stride + TransactionDT must sort by code first, then time. The stride
# must exceed every TransactionDT (train + Kaggle test reach ~3.4e7 seconds).
_KEY_STRIDE = 10**9


def window_counts(codes: np.ndarray, dt: np.ndarray, seconds: int) -> np.ndarray:
    """Per row, how many *earlier* rows with the same code fall in [dt - seconds, dt].

    ``codes`` are non-negative integer group ids (e.g. from ``ngroup()``),
    ``dt`` is TransactionDT. Rows can be in any order; the result is aligned
    with the input. Sorting by (code, time) and encoding both into one key
    lets a single binary search find every window's start at once.
    """
    dt = np.asarray(dt, dtype=np.int64)
    if len(dt) and dt.max() >= _KEY_STRIDE:
        raise ValueError(f"TransactionDT must be below {_KEY_STRIDE:,}.")

    order = np.lexsort((dt, codes))
    key = np.asarray(codes, dtype=np.int64)[order] * _KEY_STRIDE + dt[order]
    start = np.searchsorted(key, key - seconds, side="left")
    counts = np.empty(len(key), dtype=np.int64)
    counts[order] = np.arange(len(key)) - start
    return counts

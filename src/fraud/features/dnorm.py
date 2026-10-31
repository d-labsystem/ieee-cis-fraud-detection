"""D-column normalisation (Phase 4c).

Many D columns count "days since X" for a fixed X (card opened, account
created...). Raw, they grow by one every day for the same client: a time
trend that trees can't carry into later months. ``day - D`` turns them into
the fixed day X happened, which stays constant per client.

Only the columns where that holds are normalised. Measured on the full train
data (median within-client standard deviation, clients with 3+ transactions):

    anchors, normalised:  D1 D2 D4 D6 D10 D11 D12 D15  (~10-30 days -> 0-1)
                          D8                           (11 -> 3.4)
    gaps, left raw:       D3 D5 D7 D9 D13 D14          (normalising increases it)

The gaps are "days since the previous transaction"-type values: already
stable, so subtracting ``day`` would add a trend instead of removing one.
"""

import numpy as np
import pandas as pd

ANCHORS: tuple[str, ...] = ("D1", "D2", "D4", "D6", "D8", "D10", "D11", "D12", "D15")


def add_dnorm_features(df: pd.DataFrame, replace: bool = False) -> pd.DataFrame:
    """Add ``<D>n = day - <D>`` for each anchor column.

    ``D1n`` matches the client step's column of the same name. With
    ``replace=True`` the raw anchor columns are dropped, so the model only
    sees the drift-free version.
    """
    out = df.copy(deep=False)
    for col in ANCHORS:
        out[f"{col}n"] = (out["day"] - out[col]).astype(np.float32)
    if replace:
        out = out.drop(columns=list(ANCHORS))
    return out

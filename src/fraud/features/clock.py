"""Time of day (Phase 4e).

TransactionDT counts seconds from an unknown start, so ``hour`` is relative,
not local clock time. It is still consistent for every row, and that is all
a tree needs. On the train months, the fraud rate ranges from 2.2% to 10.0%
by hour, peaking in the low-volume hours (night in the main customer time
zone). The pattern holds within products (W: 1.6-4.9%, C: 4.5-39.3%) and
carries over to month 4 (rank correlation 0.91 with the train months).

Known at the moment of the transaction, so causal.
"""

import numpy as np
import pandas as pd

SECONDS_PER_HOUR = 3_600
HOURS_PER_DAY = 24


def add_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``hour`` (0-23, relative to the data's unknown start)."""
    out = df.copy(deep=False)
    out["hour"] = ((out["TransactionDT"] // SECONDS_PER_HOUR) % HOURS_PER_DAY).astype(np.float32)
    return out

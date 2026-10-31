"""Missing-value counts (Phase 4d).

XGBoost already learns a direction for NaN at every split, so per-column
"is missing" flags add nothing. What a tree can't do in one split is *count*
across columns. Columns go missing in blocks (one upstream source each), so a
per-family count tells the model which sources reported on this transaction.
"""

import numpy as np
import pandas as pd

# Raw column families, matched on the original names (so D1n, D4n... and the
# engineered features are not counted).
FAMILIES: dict[str, str] = {
    "V": r"^V\d+$",
    "D": r"^D\d+$",
    "M": r"^M\d+$",
    "id": r"^id_\d+$",
}
RAW_COLUMNS = (
    r"^[VCDM]\d+$|^id_\d+$|^(card|addr|dist)\d$|^[PR]_emaildomain$|^Device(Type|Info)$"
)


def _match(df: pd.DataFrame, pattern: str) -> list[str]:
    return list(df.columns[df.columns.str.match(pattern)])


def add_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``n_missing_<family>``, ``n_missing_raw`` and ``has_identity``.

    Must run before ``dnorm_replace``, which drops raw D columns.
    """
    out = df.copy(deep=False)
    for name, pattern in FAMILIES.items():
        out[f"n_missing_{name}"] = out[_match(out, pattern)].isna().sum(axis=1).astype(np.float32)
    out["n_missing_raw"] = out[_match(out, RAW_COLUMNS)].isna().sum(axis=1).astype(np.float32)
    # One flag for "this transaction has an identity record" (38 id_ columns).
    out["has_identity"] = out[_match(out, FAMILIES["id"])].notna().any(axis=1).astype(np.float32)
    return out

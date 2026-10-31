"""SHAP explanations for the XGBoost model (Phase 5).

A SHAP value says how much one feature moved one transaction's score, in
log-odds, relative to the average prediction. For every row:

    bias + sum(SHAP values) = the model's raw score (log-odds)
    probability = 1 / (1 + exp(-raw score))

XGBoost computes exact TreeSHAP itself (``pred_contribs=True``, fast on GPU,
handles categorical features), so no separate explainer is needed.
"""

import re

import numpy as np
import pandas as pd
import xgboost as xgb

BIAS = "bias"


def shap_values(
    model: xgb.XGBClassifier, X: pd.DataFrame, features: list[str], best_iteration: int
) -> pd.DataFrame:
    """One row per transaction, one column per feature, plus ``bias``.

    ``best_iteration`` must be passed: Booster.predict uses *all* trees by
    default, including the ones built after the best iteration.
    """
    dmatrix = xgb.DMatrix(X[features], enable_categorical=True)
    contribs = model.get_booster().predict(
        dmatrix, pred_contribs=True, iteration_range=(0, best_iteration + 1)
    )
    return pd.DataFrame(contribs, index=X.index, columns=[*features, BIAS])


# Family of each feature, checked in order; first match wins.
_FAMILIES: list[tuple[str, str]] = [
    (r"^uid_", "4a client history"),
    (r"^D\d+n$", "4c normalised D (incl. D1n)"),
    (r"_(n_24h|share_30d)$", "4b entity frequency"),
    (r"^hour$", "4e hour"),
    (r"^V\d+$", "raw V"),
    (r"^C\d+$", "raw C"),
    (r"^D\d+$", "raw D (gaps)"),
    (r"^M\d+$", "raw M"),
    (r"^id_\d+$|^Device", "raw identity/device"),
    (r"^(card|addr|dist)\d$|_emaildomain$", "raw card/address/email"),
    (r"^(TransactionAmt|ProductCD)$", "raw amount/product"),
]


def feature_family(name: str) -> str:
    for pattern, family in _FAMILIES:
        if re.search(pattern, name):
            return family
    return "other"


def global_importance(shap: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Per feature: mean |SHAP|, its share, and which way the feature pushes.

    ``direction`` is the Spearman correlation between a numeric feature's
    value and its SHAP value: +1 means "higher value -> more fraud-like",
    -1 the reverse, near 0 a non-monotonic effect. ``shap_if_missing`` is the
    mean push when the value is missing (XGBoost's learned NaN direction).
    """
    contrib = shap.drop(columns=BIAS)
    mean_abs = contrib.abs().mean()
    rows = []
    for col in contrib.columns:
        x, s = X[col], contrib[col]
        missing = x.isna().to_numpy()
        numeric = pd.api.types.is_numeric_dtype(x)
        present = ~missing
        direction = (
            pd.Series(x[present].to_numpy()).corr(pd.Series(s[present].to_numpy()), method="spearman")
            # Unused features have all-zero SHAP: no direction to measure.
            if numeric and present.sum() > 2 and x[present].nunique() > 1 and s[present].nunique() > 1
            else np.nan
        )
        rows.append({
            "feature": col,
            "family": feature_family(col),
            "mean_abs_shap": mean_abs[col],
            "direction": direction,
            "missing": missing.mean(),
            "shap_if_missing": s[missing].mean() if missing.any() else np.nan,
        })
    out = pd.DataFrame(rows).set_index("feature")
    out["share_%"] = 100 * out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    out["rank"] = out["mean_abs_shap"].rank(ascending=False).astype(int)
    return out.sort_values("mean_abs_shap", ascending=False)


def family_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Share of total mean |SHAP| per feature family."""
    return (
        importance.groupby("family")
        .agg(features=("share_%", "size"), share_pct=("share_%", "sum"))
        .sort_values("share_pct", ascending=False)
    )


def category_effects(
    shap: pd.DataFrame, X: pd.DataFrame, y: pd.Series, col: str, top: int = 10
) -> pd.DataFrame:
    """Mean push per category of a categorical feature, most common first.

    Beeswarm colours treat categories as their (alphabetical) codes, which
    means nothing; this table is the right way to read a categorical feature.
    """
    frame = pd.DataFrame({
        "value": X[col].astype(object).where(X[col].notna(), "<missing>"),
        "shap": shap[col].to_numpy(),
        "isFraud": y.to_numpy(),
    })
    out = frame.groupby("value").agg(
        transactions=("shap", "size"), fraud_rate=("isFraud", "mean"), mean_shap=("shap", "mean")
    )
    return out.sort_values("transactions", ascending=False).head(top)


def _readable(value) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)


def reasons(shap_row: pd.Series, x_row: pd.Series, top: int = 8) -> pd.DataFrame:
    """The ``top`` features that moved one transaction's score most, either way."""
    contrib = shap_row.drop(BIAS)
    order = contrib.abs().sort_values(ascending=False).index[:top]
    return pd.DataFrame({
        "value": [_readable(x_row[f]) for f in order],
        "push": contrib[order].to_numpy(),
        "direction": np.where(contrib[order] > 0, "-> more fraud-like", "-> less fraud-like"),
    }, index=pd.Index(order, name="feature"))


def pick_cases(pred: pd.DataFrame, alert_share: float) -> dict[str, int]:
    """Positions (in ``pred``) of three teaching cases.

    ``alert_share`` is the fraction of transactions an analyst team reviews;
    the threshold is the score at that cut-off.

    - caught fraud: the highest-scoring real fraud
    - false alarm:  the highest-scoring legitimate transaction
    - missed fraud: a real fraud below the threshold, at the median score
      of all missed frauds (a typical miss, not the most extreme one)
    """
    prob, fraud = pred["prob"].to_numpy(), pred["isFraud"].to_numpy().astype(bool)
    threshold = np.quantile(prob, 1 - alert_share)
    missed = np.flatnonzero(fraud & (prob < threshold))
    if not fraud.any() or fraud.all() or len(missed) == 0:
        raise ValueError("Need frauds, legitimate rows and at least one missed fraud.")
    typical_miss = missed[np.argsort(prob[missed])[len(missed) // 2]]
    return {
        "caught fraud": int(np.flatnonzero(fraud)[np.argmax(prob[fraud])]),
        "false alarm": int(np.flatnonzero(~fraud)[np.argmax(prob[~fraud])]),
        "missed fraud": int(typical_miss),
    }

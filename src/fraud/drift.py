"""Measure how the data moves over time without using labels (Phase 7).

- PSI (population stability index) per feature: how different a feature's
  distribution is now vs in training. Rule of thumb: < 0.1 stable,
  0.1-0.25 moderate shift, > 0.25 large shift.
- Adversarial validation: train a classifier to tell training rows from later
  rows. AUC ~0.5 means the periods look alike; its top features are the ones
  that drift.
- Weekly scores: ranking quality week by week, to see decay (needs labels, so
  only for the validation month).

None of these use test-month labels, so they are safe to run on the test
months (and in production, before chargebacks arrive).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from fraud.metrics import score

_MISSING, _UNSEEN = "<missing>", "<unseen>"
_FLOOR = 1e-4  # empty bins would make PSI infinite


def _numeric_bins(expected: pd.Series, actual: pd.Series, bins: int) -> tuple[pd.Series, pd.Series]:
    """Quantile bins of ``expected`` (missing values get their own bin)."""
    edges = np.unique(np.nanquantile(expected.to_numpy(dtype=float), np.linspace(0, 1, bins + 1)[1:-1]))

    def label(s: pd.Series) -> pd.Series:
        values = s.to_numpy(dtype=float)
        out = pd.Series(np.searchsorted(edges, values, side="right").astype(str), index=s.index)
        return out.where(~np.isnan(values), _MISSING)

    return label(expected), label(actual)


def _category_bins(expected: pd.Series, actual: pd.Series) -> tuple[pd.Series, pd.Series]:
    """One bin per category; categories never seen in ``expected`` share one."""
    e = expected.astype(object).where(expected.notna(), _MISSING).astype(str)
    a = actual.astype(object).where(actual.notna(), _MISSING).astype(str)
    return e, a.where(a.isin(set(e)), _UNSEEN)


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index of ``actual`` relative to ``expected``."""
    if pd.api.types.is_numeric_dtype(expected) and not isinstance(expected.dtype, pd.CategoricalDtype):
        e, a = _numeric_bins(expected, actual, bins)
    else:
        e, a = _category_bins(expected, actual)
    e_share = e.value_counts(normalize=True)
    a_share = a.value_counts(normalize=True)
    labels = e_share.index.union(a_share.index)
    e_share = e_share.reindex(labels, fill_value=0).clip(lower=_FLOOR)
    a_share = a_share.reindex(labels, fill_value=0).clip(lower=_FLOOR)
    return float(((a_share - e_share) * np.log(a_share / e_share)).sum())


def psi_table(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.Series:
    """PSI of every feature, largest first."""
    return pd.Series(
        {f: psi(reference[f], current[f]) for f in features}, name="psi"
    ).sort_values(ascending=False)


def adversarial_validation(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    device: str = "cpu",
    seed: int = 0,
    holdout: float = 0.3,
) -> tuple[float, pd.Series]:
    """Can a model tell ``current`` rows from ``reference`` rows?

    Returns the held-out ROC-AUC and each feature's share of total gain.
    """
    X = pd.concat([reference[features], current[features]], ignore_index=True)
    y = np.r_[np.zeros(len(reference)), np.ones(len(current))]
    test = np.random.default_rng(seed).random(len(X)) < holdout
    model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.5,
        tree_method="hist", enable_categorical=True, device=device, random_state=seed,
    )
    model.fit(X[~test], y[~test], verbose=False)
    auc = roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])
    gain = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
    share = (100 * gain / gain.sum()).reindex(features, fill_value=0.0)
    return float(auc), share.sort_values(ascending=False).rename("gain_%")


def week_scores(
    y: np.ndarray, prob: np.ndarray, day: np.ndarray, start: int | None = None
) -> pd.DataFrame:
    """ROC-AUC / PR-AUC per 7-day week, counted from ``start`` (default: first day given).

    Weeks with only one class (no frauds, or only frauds) get NaN scores.
    """
    y, prob, day = np.asarray(y), np.asarray(prob), np.asarray(day)
    week = (day - (day.min() if start is None else start)) // 7
    rows = {}
    for w in np.unique(week):
        mask = week == w
        s = score(y[mask], prob[mask]) if 0 < y[mask].sum() < mask.sum() else None
        rows[int(w)] = {"first_day": int(day[mask].min()), "transactions": int(mask.sum()),
                        "fraud_rate": y[mask].mean(),
                        "roc_auc": s.roc_auc if s else np.nan, "pr_auc": s.pr_auc if s else np.nan,
                        "pr_lift": s.pr_lift if s else np.nan}
    return pd.DataFrame(rows).T.rename_axis("week")

def segment_week_scores(
    y: np.ndarray, prob: np.ndarray, day: np.ndarray, segment: np.ndarray
) -> pd.DataFrame:
    """``week_scores`` per segment, plus each segment's share of the week's rows.

    Weeks count from the first day overall, so they line up across segments.
    """
    y, prob, day, segment = map(np.asarray, (y, prob, day, segment))
    start = int(day.min())
    rows_per_week = pd.Series((day - start) // 7).value_counts()
    parts = []
    for name in pd.unique(segment):
        mask = segment == name
        scores = week_scores(y[mask], prob[mask], day[mask], start=start)
        scores["share_of_week"] = scores["transactions"] / rows_per_week.reindex(scores.index)
        parts.append(scores.assign(segment=name).set_index("segment", append=True))
    return pd.concat(parts).sort_index()

"""Feature steps by name, and the final feature set chosen in Phase 4.

Features are built on the full table before splitting. That is safe only
because every step is causal (uses earlier rows only) or, like ``vreduce``,
fits on the training months only.
"""

import time
from functools import partial
from typing import Iterable

import pandas as pd

from fraud.features.client import add_client_features
from fraud.features.clock import add_hour
from fraud.features.device import add_device_features
from fraud.features.dnorm import add_dnorm_features
from fraud.features.entity import add_entity_features
from fraud.features.missing import add_missing_features
from fraud.features.vreduce import drop_correlated_v

# Steps always run in this order, whatever order they are given in. Order
# matters: "client" and "missing" read raw D columns, which "dnorm_replace"
# drops; "entity" reads raw id_30 / id_31 / DeviceInfo, which "device" drops.
STEPS = {
    "client": add_client_features,
    "entity": add_entity_features,
    "missing": add_missing_features,
    "dnorm": add_dnorm_features,
    "dnorm_replace": partial(add_dnorm_features, replace=True),
    "vreduce": drop_correlated_v,
    "hour": add_hour,
    "device": add_device_features,
}

# Phase 4 result (validation, 3 seeds): ROC-AUC 0.9308, PR-AUC 0.6361.
# "missing" was neutral and "vreduce" hurt, so neither is included.
FINAL_STEPS: tuple[str, ...] = ("client", "entity", "dnorm_replace", "hour")


def ordered_steps(steps: Iterable[str]) -> list[str]:
    """Validate step names and put them in run order."""
    steps = set(steps)
    unknown = steps - set(STEPS)
    if unknown:
        raise ValueError(f"Unknown steps: {sorted(unknown)}")
    if {"dnorm", "dnorm_replace"} <= steps:
        raise ValueError("Choose either dnorm or dnorm_replace, not both.")
    return [s for s in STEPS if s in steps]


def steps_name(steps: Iterable[str]) -> str:
    """The name a configuration is logged under, e.g. 'client+entity'."""
    return "+".join(ordered_steps(steps)) or "baseline"


def build_features(df: pd.DataFrame, steps: Iterable[str], verbose: bool = False) -> pd.DataFrame:
    for step in ordered_steps(steps):
        start = time.perf_counter()
        df = STEPS[step](df)
        if verbose:
            print(f"Step {step!r}: {time.perf_counter() - start:.1f}s")
    return df

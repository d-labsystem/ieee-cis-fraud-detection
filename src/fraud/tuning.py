"""XGBoost parameter overrides from the command line (Phase 6).

A tuning run is logged under its feature steps plus the overrides, e.g.
``client+entity@learning_rate=0.1,max_depth=6``, so ``compare`` can pair it
by seed against the untuned run like any feature experiment. "@", "," and "="
need no quoting in a shell.
"""

from typing import Iterable

from fraud.baseline import PARAMS


def _cast(key: str, text: str):
    default = PARAMS.get(key)
    if isinstance(default, bool):
        if text not in ("True", "False"):
            raise ValueError(f"{key} must be True or False, got {text!r}")
        return text == "True"
    if default is not None:
        return type(default)(text)
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_params(items: Iterable[str]) -> dict:
    """``["learning_rate=0.1", "max_depth=6"]`` -> ``{"learning_rate": 0.1, "max_depth": 6}``.

    Values take the type of the default in PARAMS; keys not in PARAMS (any
    other XGBoost parameter, e.g. min_child_weight) become int, float or str.
    """
    params = {}
    for item in items:
        key, sep, text = item.partition("=")
        if not sep or not key or not text:
            raise ValueError(f"Expected key=value, got {item!r}")
        params[key] = _cast(key, text)
    return params


def config_name(steps: str, params: dict) -> str:
    """Log name: the steps alone, or the steps plus sorted overrides."""
    if not params:
        return steps
    return steps + "@" + ",".join(f"{k}={v}" for k, v in sorted(params.items()))

"""Version-robust browser, OS and device features (Phase 4f).

``id_31`` (browser), ``id_30`` (OS) and ``DeviceInfo`` contain version
strings that go stale: "chrome 66.0" has 0 rows in the train months and
4,264 in the test months, where 43.5% of identity rows carry a browser
string the model never saw. A tree can't tell that chrome 66 is just a newer
chrome, so each new release gets a generic treatment.

Each column is split into a stable family (categorical) and a version
(numeric, so a split like "version >= 65" still applies to 66), plus a causal
"versions behind the newest seen so far" that stays meaningful as releases
continue. The raw columns are dropped.

Versions are encoded as major + minor / 100, so 10.9 -> 10.09 sorts below
10.12 -> 10.12 (as plain decimals 10.9 > 10.12 would be wrong).
"""

import re

import numpy as np
import pandas as pd

RAW_COLUMNS: tuple[str, ...] = ("id_31", "id_30", "DeviceInfo")

_VERSION = re.compile(r"(\d+)(?:[._](\d+))?")

# DeviceInfo -> brand, checked in order on the lower-cased value.
_BRANDS: list[tuple[str, str]] = [
    (r"^windows$", "windows"),
    (r"^ios device$|^macos$|iphone|ipad", "apple"),
    (r"^trident/", "windows"),  # Internet Explorer's engine token
    (r"^rv:", "firefox"),  # Firefox's version token, no device given
    (r"^sm-|^samsung|^gt-|^sgh-|^sch-", "samsung"),
    (r"^moto|^xt\d", "motorola"),
    (r"^lg|^vs\d", "lg"),
    (r"huawei|^ale-|^cam-|^pra-|^hi\d|^[a-z]{3}-l\d\d", "huawei"),
    (r"redmi|^mi |xiaomi", "xiaomi"),
    (r"^pixel|^nexus", "google"),
    (r"zte|^blade", "zte"),
    (r"asus", "asus"),
    (r"lenovo", "lenovo"),
    (r"^kf", "amazon"),
    (r"^htc", "htc"),
    (r"^[a-z]\d{4}\b|^sony|xperia", "sony"),
]


def encode_version(text) -> float:
    """First 'major[.minor]' in ``text`` as major + minor/100; NaN if none."""
    if not isinstance(text, str):
        return np.nan
    m = _VERSION.search(text)
    if m is None:
        return np.nan
    return int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 100


def browser_family(text) -> str | float:
    """'chrome 65.0 for android' -> 'chrome'; 'mobile safari generic' -> 'mobile safari'."""
    if not isinstance(text, str):
        return np.nan
    s = text.lower()
    s = re.sub(r"\s+for\s+\w+$", "", s)  # platform suffix; DeviceType covers it
    s = s.split("/")[0]  # 'samsung/sm-g532m' -> 'samsung'
    s = re.sub(r"\bgeneric\b", "", s)
    s = re.sub(r"\d+(\.\d+)*", "", s)
    return s.strip() or "other"


def os_family(text) -> str | float:
    """'Mac OS X 10_12_6' -> 'mac'; 'iOS 11.2.1' -> 'ios'; 'Windows 10' -> 'windows'."""
    if not isinstance(text, str):
        return np.nan
    s = text.lower()
    for prefix in ("windows", "ios", "android", "linux", "mac", "func", "other"):
        if s.startswith(prefix):
            return prefix
    return "other"


def device_brand(text) -> str | float:
    if not isinstance(text, str):
        return np.nan
    s = text.lower()
    for pattern, brand in _BRANDS:
        if re.search(pattern, s):
            return brand
    return "other"


def device_model(text) -> str | float:
    """'SM-G930V Build/NRD90M' -> 'SM-G930V': drop the firmware build, which goes stale."""
    if not isinstance(text, str):
        return np.nan
    model = text.split(" Build/")[0].strip()
    return model if model and not model.startswith("rv:") else np.nan


def versions_behind(family: pd.Series, version: pd.Series, dt: pd.Series) -> np.ndarray:
    """Newest version of this family seen so far (this row included) minus this
    row's version, in time order. Causal: uses only earlier and current rows."""
    order = np.argsort(dt.to_numpy(), kind="stable")
    fam = family.to_numpy(dtype=object)[order]
    ver = version.to_numpy(dtype=np.float64)[order]
    newest = pd.Series(ver).groupby(pd.Series(fam).fillna("<na>").to_numpy()).cummax().to_numpy()
    behind_sorted = newest - ver
    behind = np.empty_like(behind_sorted)
    behind[order] = behind_sorted
    behind[pd.isna(family.to_numpy())] = np.nan
    return behind


def _categorical(values: pd.Series) -> pd.Series:
    return values.astype("category")


def add_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace id_31 / id_30 / DeviceInfo with family, version and age features."""
    out = df.copy(deep=False)
    raw = {c: out[c].astype(object).where(out[c].notna(), None) for c in RAW_COLUMNS}

    browser = raw["id_31"].map(browser_family)
    browser_version = raw["id_31"].map(encode_version).astype(np.float64)
    os_ = raw["id_30"].map(os_family)
    os_version = raw["id_30"].map(encode_version).astype(np.float64)

    out["browser"] = _categorical(browser)
    out["browser_version"] = browser_version.astype(np.float32)
    out["browser_versions_behind"] = versions_behind(
        browser, browser_version, out["TransactionDT"]
    ).astype(np.float32)
    out["os"] = _categorical(os_)
    out["os_version"] = os_version.astype(np.float32)
    out["os_versions_behind"] = versions_behind(os_, os_version, out["TransactionDT"]).astype(
        np.float32
    )
    out["device_brand"] = _categorical(raw["DeviceInfo"].map(device_brand))
    out["device_model"] = _categorical(raw["DeviceInfo"].map(device_model))
    return out.drop(columns=list(RAW_COLUMNS))

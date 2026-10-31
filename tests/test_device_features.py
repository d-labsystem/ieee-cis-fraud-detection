import numpy as np
import pandas as pd
import pytest

from fraud.features.device import (
    add_device_features,
    browser_family,
    device_brand,
    device_model,
    encode_version,
    os_family,
)


@pytest.mark.parametrize(
    ("text", "version"),
    [
        ("chrome 65.0", 65.0),
        ("samsung browser 6.4", 6.04),
        ("iOS 11.2.1", 11.02),
        ("Mac OS X 10_12_6", 10.12),
        ("Mac OS X 10_9_5", 10.09),  # sorts below 10.12, unlike the decimal 10.9
        ("Windows 10", 10.0),
        ("mobile safari generic", np.nan),
        (None, np.nan),
    ],
)
def test_encode_version(text, version):
    np.testing.assert_equal(encode_version(text), version)


@pytest.mark.parametrize(
    ("text", "family"),
    [
        ("chrome 65.0 for android", "chrome"),
        ("chrome generic for android", "chrome"),
        ("mobile safari 11.0", "mobile safari"),
        ("safari generic", "safari"),
        ("ie 11.0 for desktop", "ie"),
        ("samsung browser 6.2", "samsung browser"),
        ("Samsung/SM-G532M", "samsung"),
        ("other", "other"),
    ],
)
def test_browser_family(text, family):
    assert browser_family(text) == family


@pytest.mark.parametrize(
    ("text", "family"),
    [("Windows 10", "windows"), ("iOS 11.2.1", "ios"), ("Mac OS X 10_12_6", "mac"),
     ("Mac", "mac"), ("Android 7.0", "android"), ("Linux", "linux")],
)
def test_os_family(text, family):
    assert os_family(text) == family


@pytest.mark.parametrize(
    ("text", "brand", "model"),
    [
        ("SM-G930V Build/NRD90M", "samsung", "SM-G930V"),
        ("Moto G (4) Build/NPJ25.93-14.7", "motorola", "Moto G (4)"),
        ("ALE-L23 Build/HuaweiALE-L23", "huawei", "ALE-L23"),
        ("iOS Device", "apple", "iOS Device"),
        ("Trident/7.0", "windows", "Trident/7.0"),
        ("rv:57.0", "firefox", np.nan),
        ("Kylin", "other", "Kylin"),
    ],
)
def test_device_brand_and_model(text, brand, model):
    assert device_brand(text) == brand
    np.testing.assert_equal(device_model(text), model)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionDT": [100, 200, 300, 400, 500],
            "id_31": ["chrome 63.0", "chrome 65.0", "chrome 63.0", "safari generic", None],
            "id_30": ["Windows 10", "iOS 11.2.1", "iOS 11.3.0", "iOS 11.2.1", None],
            "DeviceInfo": ["Windows", "iOS Device", "iOS Device", None, None],
        }
    )


def test_replaces_raw_columns():
    out = add_device_features(_df())
    assert not {"id_31", "id_30", "DeviceInfo"} & set(out.columns)
    assert out["browser"].tolist()[:4] == ["chrome", "chrome", "chrome", "safari"]
    assert isinstance(out["browser"].dtype, pd.CategoricalDtype)
    assert pd.isna(out["browser"].iloc[4])


def test_versions_behind_is_causal():
    out = add_device_features(_df())
    # chrome 63 first (newest so far -> 0), then 65 (0), then 63 again (65 - 63 = 2).
    np.testing.assert_allclose(out["browser_versions_behind"].iloc[:3], [0, 0, 2])
    # iOS: 11.02, then 11.03 at t=300, then 11.02 at t=400 -> 0.01 behind.
    np.testing.assert_allclose(out["os_versions_behind"].iloc[1:4], [0, 0, 0.01], atol=1e-6)
    assert out["browser_versions_behind"].iloc[3:].isna().all()  # no version / no browser


def test_versions_behind_any_row_order():
    expected = add_device_features(_df())
    shuffled = add_device_features(_df().iloc[[4, 2, 0, 3, 1]])
    pd.testing.assert_series_equal(
        shuffled["browser_versions_behind"].sort_index(), expected["browser_versions_behind"]
    )

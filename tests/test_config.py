from pathlib import Path

import pytest
from pydantic import ValidationError

from fraud.config import (
    KAGGLE_INPUT_DIR,
    KAGGLE_WORKING_DIR,
    PROJECT_ROOT,
    Settings,
    get_settings,
KAGGLE_ENV_VAR
)

# Every environment variable that can change Settings. Removed before each
# test so a value set in your shell can't leak in and change the result.
_SETTINGS_ENV_VARS = (
    "PROJECT_ROOT",
    "RAW_DIR",
    "SAMPLE_DIR",
    "PROCESSED_DIR",
    "KAGGLE_ENV_VAR",
)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    for var in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_project_root_contains_pyproject():
    assert (get_settings().project_root / "pyproject.toml").is_file()


def test_local_defaults():
    settings = get_settings()
    assert settings.raw_dir == PROJECT_ROOT / "data" / "raw" / "ieee-fraud-detection"
    assert settings.sample_dir == PROJECT_ROOT / "data" / "sample"
    assert settings.processed_dir == PROJECT_ROOT / "data" / "processed"


def test_kaggle_defaults(monkeypatch):
    monkeypatch.setenv(KAGGLE_ENV_VAR, "Interactive")
    settings = get_settings()
    assert settings.raw_dir == KAGGLE_INPUT_DIR
    assert settings.processed_dir == KAGGLE_WORKING_DIR


def test_env_var_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("RAW_DIR", str(tmp_path))
    assert get_settings().raw_dir == tmp_path


def test_env_var_overrides_kaggle_default(monkeypatch, tmp_path):
    monkeypatch.setenv(KAGGLE_ENV_VAR, "Interactive")
    monkeypatch.setenv("RAW_DIR", str(tmp_path))
    assert get_settings().raw_dir == tmp_path


def test_constructor_accepts_field_names(tmp_path):
    assert Settings(raw_dir=tmp_path).raw_dir == tmp_path


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        Settings(raw_dirr=Path("/typo"))


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
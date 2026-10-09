"""Project configuration: where data lives, on a Mac and on Kaggle.

Resolution order for every setting (highest priority first):
    1. Arguments passed to ``Settings(...)`` directly (used in tests).
    2. Environment variables, matched case-insensitively (``RAW_DIR``).
    3. A ``.env`` file in the project root, if one exists.
    4. The defaults defined below.
"""

import os
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# code that runs on Kaggle must never use paths derived from PROJECT_ROOT.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

KAGGLE_INPUT_DIR: Path = Path("/kaggle/input/ieee-fraud-detection")
KAGGLE_WORKING_DIR: Path = Path("/kaggle/working")

def _on_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_ID" in os.environ

def _default_raw_dir() -> Path:
    if _on_kaggle():
        return KAGGLE_INPUT_DIR
    return PROJECT_ROOT / "data" / "raw" / "ieee-fraud-detection"


def _default_processed_dir() -> Path:
    if _on_kaggle():
        return KAGGLE_WORKING_DIR
    return PROJECT_ROOT / "data" / "processed"


def _default_sample_dir() -> Path:
    return PROJECT_ROOT / "data" / "sample"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT/ ".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    project_root: Path = PROJECT_ROOT
    raw_dir: Path =  Field(default_factory=_default_raw_dir)
    sample_dir: Path =  Field(default_factory=_default_sample_dir)
    processed_dir: Path =  Field(default_factory=_default_processed_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
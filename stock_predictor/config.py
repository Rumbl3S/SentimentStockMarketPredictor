"""Configuration and environment loading for stock predictor."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_TOP_K = 4
HISTORICAL_DAYS = 730
LOCAL_DATASET_PATH = BASE_DIR / "data" / "final" / "final_model_dataset.csv"

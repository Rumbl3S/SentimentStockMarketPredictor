"""Configuration and environment loading for stock predictor."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MARKETAUX_BASE_URL = "https://api.marketaux.com/v1"
DEFAULT_TOP_K = 5
HISTORICAL_DAYS = 730
FREE_TIER_LIMIT_PER_REQUEST = 3
FREE_TIER_DAILY_REQUESTS = 100

# Accept both names so users can drop in either key name.
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY") or os.getenv("NEWS_API", "")

"""Local dataset news retrieval (no MarketAux dependency)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

API_REQUEST_COUNT = 0
LOCAL_NEWS_PATH = (
    Path(__file__).resolve().parent / "data" / "processed" / "news_articles_mapped.csv"
)


def get_api_request_count() -> int:
    # Kept for interface compatibility; local mode uses no API requests.
    return API_REQUEST_COUNT


def reset_api_request_count() -> None:
    global API_REQUEST_COUNT
    API_REQUEST_COUNT = 0


def fetch_articles_for_ticker(
    ticker: str, search_keywords: list[str], num_pages: int = 3, top_k: int = 3
) -> list[dict[str, Any]]:
    """Fetch ticker articles from local processed dataset."""
    if not LOCAL_NEWS_PATH.exists():
        print(f"Warning: local news dataset not found at {LOCAL_NEWS_PATH}")
        return []

    normalized_ticker = ticker.upper().strip()
    news = pd.read_csv(LOCAL_NEWS_PATH)
    if news.empty:
        return []

    required_cols = {"ticker", "title", "url", "source", "date"}
    if not required_cols.issubset(set(news.columns)):
        print("Warning: local news dataset is missing required columns.")
        return []

    df = news[news["ticker"].astype(str).str.upper() == normalized_ticker].copy()
    if df.empty:
        return []

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date", ascending=False)

    # Soft keyword filter first, then fallback to most recent items if too few.
    keywords = [k.lower() for k in search_keywords if k]
    if keywords:
        text_blob = (
            df["title"].fillna("")
            + " "
            + df.get("description", pd.Series("", index=df.index)).fillna("")
            + " "
            + df.get("highlight", pd.Series("", index=df.index)).fillna("")
        ).str.lower()

        def _kw_score(text: str) -> float:
            if not keywords:
                return 0.0
            hits = sum(1 for kw in keywords if kw in text)
            return hits / max(1, len(keywords))

        df["keyword_match_score"] = text_blob.map(_kw_score)
        narrowed = df[df["keyword_match_score"] > 0]
    else:
        df["keyword_match_score"] = 0.0
        narrowed = df

    # Stage 1 retrieval: intentionally broad candidate pool.
    limit_rows = max(120, top_k * max(1, num_pages) * 20)
    if len(narrowed) >= top_k:
        selected = narrowed.sort_values(
            ["keyword_match_score", "date"], ascending=[False, False]
        ).head(limit_rows)
    else:
        selected = df.sort_values("date", ascending=False).head(limit_rows)

    out: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        published = row["date"].isoformat() if isinstance(row["date"], datetime) else str(row["date"])
        out.append(
            {
                "uuid": f"local::{normalized_ticker}::{hash((row.get('url', ''), row.get('title', '')))}",
                "ticker": normalized_ticker,
                "title": str(row.get("title", "")),
                "description": "",
                "description": str(row.get("description", "")),
                "snippet": str(row.get("highlight", "")),
                "url": str(row.get("url", "")),
                "source": str(row.get("source", "local_dataset")),
                "published_at": published,
                "entity_sentiment_score": float(row.get("sentiment_score", 0.0)),
                "entity_match_score": float(row.get("keyword_match_score", 0.0)),
                "highlights": [
                    {
                        "highlight": str(row.get("highlight", "")),
                        "sentiment": float(row.get("sentiment_score", 0.0)),
                        "highlighted_in": "main_text",
                    }
                ],
            }
        )

    return out

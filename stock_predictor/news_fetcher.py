"""Local dataset news retrieval (no MarketAux dependency). Polars I/O.

Falls back to live Google News RSS for the requested ticker when the local
mapped CSV has no rows for that symbol (e.g. ticker was not in the dataset
build universe, or the last build predates caring about that name).
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

API_REQUEST_COUNT = 0
LOCAL_NEWS_PATH = (
    Path(__file__).resolve().parent / "data" / "processed" / "news_articles_mapped.csv"
)

# Lazy import target for RSS reuse (same queries as dataset build).
_build_rss_module: Any = None


def _get_rss_for_ticker():
    global _build_rss_module
    if _build_rss_module is None:
        from build_local_dataset import rss_for_ticker as _rf

        _build_rss_module = _rf
    return _build_rss_module


def _live_rss_fallback_frame(normalized_ticker: str, lookback_days: int = 3650) -> pl.DataFrame:
    """Pull fresh RSS rows for one ticker; shape matches mapped CSV expectations."""
    rss_for_ticker = _get_rss_for_ticker()
    raw_rows = rss_for_ticker(normalized_ticker, lookback_days=lookback_days)
    if not raw_rows:
        return pl.DataFrame()

    seen: set[tuple[str, str]] = set()
    mapped: list[dict[str, Any]] = []
    for r in raw_rows:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        key = (url, title)
        if not title or key in seen:
            continue
        seen.add(key)
        desc = (r.get("description") or "").strip()
        blob = f"{title}. {desc}"[:500]
        mapped.append(
            {
                "ticker": normalized_ticker,
                "title": title,
                "description": desc,
                "highlight": blob[:280],
                "url": url,
                "source": (r.get("source") or "google_rss_live").strip() or "google_rss_live",
                "published_at": (r.get("published_at") or "").strip(),
                "sentiment_score": 0.0,
            }
        )
    if not mapped:
        return pl.DataFrame()

    fb = pl.DataFrame(mapped)
    pub = pl.col("published_at")
    # Polars requires time_zone when strings include offsets; coalesce handles plain dates.
    date_from_iso = pub.str.to_datetime(time_zone="UTC", strict=False).dt.date()
    date_from_prefix = pub.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
    return fb.with_columns(
        pl.when(pub.str.len_chars() > 8)
        .then(pl.coalesce(date_from_iso, date_from_prefix))
        .otherwise(None)
        .alias("date_parsed")
    ).drop_nulls("date_parsed")


def get_api_request_count() -> int:
    return API_REQUEST_COUNT


def reset_api_request_count() -> None:
    global API_REQUEST_COUNT
    API_REQUEST_COUNT = 0


def fetch_articles_for_ticker(
    ticker: str, search_keywords: list[str], num_pages: int = 3, top_k: int = 3
) -> list[dict[str, Any]]:
    """Load articles for a ticker from local CSV, or live Google News RSS if none are stored."""
    normalized_ticker = ticker.upper().strip()
    df = pl.DataFrame()

    required = {"ticker", "title", "url", "source", "date"}
    if LOCAL_NEWS_PATH.exists():
        news = pl.read_csv(LOCAL_NEWS_PATH, try_parse_dates=True)
        if not news.is_empty() and required.issubset(set(news.columns)):
            df = news.filter(pl.col("ticker").cast(pl.Utf8).str.to_uppercase() == normalized_ticker)
        elif not news.is_empty():
            print("Warning: local news dataset is missing required columns; using RSS fallback only.")

    if df.is_empty():
        df = _live_rss_fallback_frame(normalized_ticker)
        if df.is_empty():
            if not LOCAL_NEWS_PATH.exists():
                print(f"Warning: no local news at {LOCAL_NEWS_PATH} and RSS returned no items for {normalized_ticker}.")
            return []

    if "date_parsed" not in df.columns:
        if "date" not in df.columns:
            return []
        dtype = df.schema["date"]
        if dtype == pl.Utf8:
            df = df.with_columns(pl.col("date").str.slice(0, 10).str.to_date(strict=False).alias("date_parsed"))
        else:
            df = df.with_columns(pl.col("date").cast(pl.Date, strict=False).alias("date_parsed"))
    df = df.drop_nulls("date_parsed").sort("date_parsed", descending=True)

    keywords = [k.lower() for k in search_keywords if k]
    if keywords:
        title_c = pl.col("title").cast(pl.Utf8).fill_null("")
        desc_c = pl.col("description").cast(pl.Utf8).fill_null("") if "description" in df.columns else pl.lit("")
        hl_c = pl.col("highlight").cast(pl.Utf8).fill_null("") if "highlight" in df.columns else pl.lit("")
        text_blob = (title_c + pl.lit(" ") + desc_c + pl.lit(" ") + hl_c).str.to_lowercase()

        def _kw_score_expr() -> pl.Expr:
            hits = pl.sum_horizontal(*[text_blob.str.contains(kw, literal=True) for kw in keywords])
            return hits.cast(pl.Float64) / max(1, len(keywords))

        df = df.with_columns(_kw_score_expr().alias("keyword_match_score"))
        narrowed = df.filter(pl.col("keyword_match_score") > 0)
    else:
        df = df.with_columns(pl.lit(0.0).alias("keyword_match_score"))
        narrowed = df

    limit_rows = max(120, top_k * max(1, num_pages) * 20)
    if narrowed.height >= top_k:
        selected = narrowed.sort(["keyword_match_score", "date_parsed"], descending=[True, True]).head(limit_rows)
    else:
        selected = df.sort("date_parsed", descending=True).head(limit_rows)

    out: list[dict[str, Any]] = []
    for row in selected.iter_rows(named=True):
        d = row.get("date_parsed")
        if isinstance(d, datetime):
            published = d.date().isoformat()
        elif isinstance(d, date):
            published = d.isoformat()
        else:
            published = str(row.get("date", ""))[:10]

        url = str(row.get("url", "") or "")
        title = str(row.get("title", "") or "")
        out.append(
            {
                "uuid": f"local::{normalized_ticker}::{hash((url, title))}",
                "ticker": normalized_ticker,
                "title": title,
                "description": str(row.get("description", "") or ""),
                "snippet": str(row.get("highlight", "") or ""),
                "url": url,
                "source": str(row.get("source", "local_dataset") or ""),
                "published_at": published,
                "entity_sentiment_score": float(row.get("sentiment_score", 0.0) or 0.0),
                "entity_match_score": float(row.get("keyword_match_score", 0.0) or 0.0),
                "highlights": [
                    {
                        "highlight": str(row.get("highlight", "") or ""),
                        "sentiment": float(row.get("sentiment_score", 0.0) or 0.0),
                        "highlighted_in": "main_text",
                    }
                ],
            }
        )

    return out

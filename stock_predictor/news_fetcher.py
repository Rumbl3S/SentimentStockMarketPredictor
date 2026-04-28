"""MarketAux API integration for article retrieval."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import requests

from config import (
    FREE_TIER_LIMIT_PER_REQUEST,
    MARKETAUX_API_KEY,
    MARKETAUX_BASE_URL,
)

API_REQUEST_COUNT = 0


def _published_after_iso(days_back: int = 30) -> str:
    target = datetime.utcnow() - timedelta(days=days_back)
    return target.strftime("%Y-%m-%dT%H:%M")


def get_api_request_count() -> int:
    return API_REQUEST_COUNT


def fetch_articles_for_ticker(
    ticker: str, search_keywords: list[str], num_pages: int = 3, top_k: int = 3
) -> list[dict[str, Any]]:
    """Fetch MarketAux articles with two-pass recall strategy for one ticker."""
    global API_REQUEST_COUNT
    if not MARKETAUX_API_KEY:
        print("Warning: MARKETAUX_API_KEY/NEWS_API missing. Cannot fetch news.")
        return []

    endpoint = f"{MARKETAUX_BASE_URL}/news/all"
    search_text = " ".join(search_keywords).strip()
    normalized_ticker = ticker.upper().strip()
    all_articles: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()

    def _fetch_page(page: int, use_search: bool) -> list[dict[str, Any]] | None:
        nonlocal all_articles
        params = {
            "api_token": MARKETAUX_API_KEY,
            "symbols": normalized_ticker,
            "filter_entities": "true",
            "must_have_entities": "true",
            "language": "en",
            "published_after": _published_after_iso(30),
            "group_similar": "true",
            "limit": FREE_TIER_LIMIT_PER_REQUEST,
            "page": page,
        }
        if use_search and search_text:
            params["search"] = search_text
        try:
            response = requests.get(endpoint, params=params, timeout=20)
            # both passes count toward daily budget
            global API_REQUEST_COUNT
            API_REQUEST_COUNT += 1

            if response.status_code == 429:
                mode = "targeted" if use_search else "fallback"
                print(f"  Rate limit hit for {normalized_ticker} {mode} page {page}.")
                return None
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            mode = "targeted" if use_search else "fallback"
            print(f"  Warning: failed fetching {normalized_ticker} {mode} page {page}: {exc}")
            return None
        except ValueError:
            mode = "targeted" if use_search else "fallback"
            print(f"  Warning: non-JSON response for {normalized_ticker} {mode} page {page}.")
            return None

        items = payload.get("data", [])
        mode_label = "targeted" if use_search else "fallback"
        print(f"  Fetching {mode_label} page {page}/{num_pages}... ({len(items)} articles)")
        if not items:
            return []

        for item in items:
            article_uuid = item.get("uuid", "")
            if article_uuid and article_uuid in seen_uuids:
                continue

            entity_for_ticker = None
            for ent in item.get("entities", []):
                if (ent.get("symbol") or "").upper() == normalized_ticker:
                    entity_for_ticker = ent
                    break

            highlights: list[dict[str, Any]] = []
            if entity_for_ticker:
                for h in entity_for_ticker.get("highlights", []):
                    highlights.append(
                        {
                            "highlight": h.get("highlight", ""),
                            "sentiment": h.get("sentiment"),
                            "highlighted_in": h.get("highlighted_in", ""),
                        }
                    )

            mapped = {
                "uuid": article_uuid,
                "ticker": normalized_ticker,
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "published_at": item.get("published_at", ""),
                "entity_sentiment_score": (
                    entity_for_ticker.get("sentiment_score")
                    if entity_for_ticker
                    else None
                ),
                "entity_match_score": (
                    entity_for_ticker.get("match_score") if entity_for_ticker else None
                ),
                "highlights": highlights,
            }
            all_articles.append(mapped)
            if article_uuid:
                seen_uuids.add(article_uuid)

        time.sleep(1)
        return items

    # Pass 1: targeted fetch with search + symbol.
    for page in range(1, max(1, num_pages) + 1):
        fetched = _fetch_page(page=page, use_search=True)
        if fetched is None or not fetched:
            break

    # Pass 2: fallback broadening with symbol only, if needed.
    if len(all_articles) < max(1, top_k):
        print(
            f"  Pass 1 returned {len(all_articles)} articles; broadening with ticker-only fetch."
        )
        for page in range(1, max(1, num_pages) + 1):
            if len(all_articles) >= max(1, top_k):
                break
            fetched = _fetch_page(page=page, use_search=False)
            if fetched is None or not fetched:
                break

    return all_articles

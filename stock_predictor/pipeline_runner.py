"""Shared pipeline orchestration for CLI and API interfaces."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from config import DEFAULT_TOP_K, HISTORICAL_DAYS
from news_fetcher import (
    fetch_articles_for_ticker,
    get_api_request_count,
    reset_api_request_count,
)
from predictor import StockPredictor
from price_fetcher import fetch_price_data
from query_relevance import (
    build_api_search_keywords,
    cluster_articles,
    extract_keywords,
    rank_articles_by_relevance,
)
from sentiment_analyzer import SentimentAnalyzer


def _to_json_safe(value: Any) -> Any:
    """Recursively convert common non-JSON types to plain Python values."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return value
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def run_pipeline(
    query: str,
    tickers: list[str],
    pages: int = 3,
    history_days: int = HISTORICAL_DAYS,
    top_k: int = DEFAULT_TOP_K,
    cluster: bool = False,
) -> dict[str, Any]:
    """Run full stock pipeline and return JSON-safe structured output."""
    normalized_tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not query.strip() or not normalized_tickers:
        raise ValueError("Query and at least one ticker are required.")

    reset_api_request_count()
    warnings: list[str] = []
    keywords = extract_keywords(query)
    api_keywords = build_api_search_keywords(keywords)

    analyzer = SentimentAnalyzer()
    predictor = StockPredictor()
    per_ticker: dict[str, dict[str, Any]] = {}
    combined_articles: list[dict[str, Any]] = []

    for ticker in normalized_tickers:
        articles = fetch_articles_for_ticker(
            ticker=ticker,
            search_keywords=api_keywords,
            num_pages=pages,
            top_k=top_k,
        )
        relevant_articles = rank_articles_by_relevance(query, articles, top_k=top_k)
        sentiment_results = analyzer.analyze_articles(relevant_articles)

        try:
            price_data = fetch_price_data(ticker, period_days=history_days)
        except Exception as exc:
            warnings.append(f"{ticker}: price fetch failed: {exc}")
            continue

        prediction = predictor.run_prediction(price_data, sentiment_results)
        price_payload = dict(price_data)
        price_payload.pop("history_df", None)

        per_ticker[ticker] = _to_json_safe(
            {
                "articles": relevant_articles,
                "sentiment": sentiment_results,
                "price": price_payload,
                "prediction": prediction,
            }
        )
        combined_articles.extend(sentiment_results.get("articles", []))

    cluster_info: dict[str, Any] | None = None
    if cluster and combined_articles:
        cluster_count = min(max(3, len(normalized_tickers)), len(combined_articles))
        cluster_info = _to_json_safe(
            cluster_articles(query, combined_articles, n_clusters=cluster_count)
        )

    return {
        "inputs": {
            "query": query,
            "tickers": normalized_tickers,
            "pages": int(pages),
            "history_days": int(history_days),
            "top_k": int(top_k),
            "cluster": bool(cluster),
            "keywords": keywords,
            "api_keywords": api_keywords,
        },
        "per_ticker": per_ticker,
        "cluster_info": cluster_info,
        "meta": {
            "api_requests_used": get_api_request_count(),
            "warnings": warnings,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

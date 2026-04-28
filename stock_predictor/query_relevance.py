"""Query keyword extraction, relevance ranking, and optional clustering."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GENERIC_FINANCE_TERMS = {
    "stock",
    "stocks",
    "market",
    "markets",
    "trading",
    "trade",
    "invest",
    "investment",
    "price",
    "prices",
    "affect",
    "effect",
    "impact",
    "demand",
    "supply",
    "growth",
    "decline",
    "rise",
    "fall",
    "buy",
    "sell",
    "bullish",
    "bearish",
    "forecast",
    "predict",
    "analysis",
    "financial",
}


def extract_keywords(query: str) -> list[str]:
    """Remove stopwords and return useful query keywords."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", query.lower())
    keywords = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    if not keywords:
        return tokens[:4]
    # Keep a richer keyword set for local TF-IDF re-ranking.
    return list(dict.fromkeys(keywords))[:12]


def build_api_search_keywords(keywords: list[str], min_terms: int = 2, max_terms: int = 3) -> list[str]:
    """Pick specific terms for MarketAux search; exclude generic finance noise."""
    deduped = [k for k in dict.fromkeys(keywords) if k]
    specific = [k for k in deduped if k not in GENERIC_FINANCE_TERMS]
    if len(specific) >= min_terms:
        return specific[:max_terms]
    # Fallback if query is mostly generic terms.
    return deduped[:max_terms]


def _article_text(article: dict[str, Any]) -> str:
    highlight_text = " ".join(
        str(h.get("highlight", "")) for h in article.get("highlights", [])
    )
    return " ".join(
        [
            article.get("title", ""),
            article.get("description", ""),
            article.get("snippet", ""),
            highlight_text,
        ]
    ).strip()


def rank_articles_by_relevance(
    query: str, articles: list[dict[str, Any]], top_k: int = 5
) -> list[dict[str, Any]]:
    """Rank articles by TF-IDF cosine similarity to query."""
    if not articles:
        return []

    corpus = [query] + [_article_text(article) for article in articles]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    query_vector = matrix[0:1]
    article_vectors = matrix[1:]
    similarities = cosine_similarity(query_vector, article_vectors).flatten()

    ranked = []
    for article, sim in zip(articles, similarities):
        article_copy = dict(article)
        article_copy["relevance_score"] = float(sim)
        ranked.append(article_copy)

    ranked.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    return ranked[: max(1, top_k)]


def cluster_articles(
    query: str, all_articles: list[dict[str, Any]], n_clusters: int = 3
) -> dict[str, Any]:
    """Optional K-Means clustering for exploratory topic grouping."""
    if not all_articles:
        return {"enabled": True, "clusters": {}, "query_cluster": None}

    docs = [_article_text(a) for a in all_articles]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1500)
    x = vectorizer.fit_transform(docs)

    clusters = max(1, min(n_clusters, len(all_articles)))
    model = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    labels = model.fit_predict(x)

    feature_names = np.array(vectorizer.get_feature_names_out())
    top_terms_by_cluster: dict[int, list[str]] = {}
    for cluster_id, center in enumerate(model.cluster_centers_):
        top_indices = center.argsort()[-8:][::-1]
        top_terms_by_cluster[cluster_id] = feature_names[top_indices].tolist()

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for article, label in zip(all_articles, labels):
        article_copy = dict(article)
        article_copy["cluster"] = int(label)
        grouped[int(label)].append(article_copy)

    query_vec = vectorizer.transform([query])
    query_cluster = int(model.predict(query_vec)[0])

    clusters_out: dict[int, dict[str, Any]] = {}
    for cluster_id, article_group in grouped.items():
        tickers = Counter(a.get("ticker", "UNK") for a in article_group)
        clusters_out[cluster_id] = {
            "size": len(article_group),
            "top_terms": top_terms_by_cluster.get(cluster_id, []),
            "top_tickers": tickers.most_common(3),
        }

    return {
        "enabled": True,
        "query_cluster": query_cluster,
        "clusters": clusters_out,
        "articles_with_clusters": [a for group in grouped.values() for a in group],
    }

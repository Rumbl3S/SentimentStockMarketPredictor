"""Generate synthetic but realistic test data for analysis plots.

This script creates:
- article-level records with sentiment/similarity/embeddings/clusters
- event+ticker score records for semantic vs sentiment analysis
- random-forest feature importance records
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS = [
    "AI chip demand surge",
    "Fed rates hold",
    "Geopolitical oil disruption",
    "Cloud earnings beat",
    "Banking regulation proposal",
    "EV supply chain slowdown",
    "Pharma trial results",
    "Consumer spending decline",
    "Cybersecurity breach cycle",
    "Tariff escalation talks",
    "Labor market surprise",
    "Semiconductor export controls",
]

TICKERS = ["NVDA", "TSM", "AMD", "AAPL", "MSFT", "XOM", "JPM", "TSLA", "GOOGL", "CVX"]

FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "volume_change_5d",
    "sma_10_gap",
    "sma_30_gap",
    "benchmark_return_5d",
    "relative_strength_5d",
]

# Per-event directional drift to create a realistic mix of bullish/bearish contexts.
EVENT_SENTIMENT_DRIFT = {
    "AI chip demand surge": 0.24,
    "Fed rates hold": 0.02,
    "Geopolitical oil disruption": -0.14,
    "Cloud earnings beat": 0.19,
    "Banking regulation proposal": -0.18,
    "EV supply chain slowdown": -0.20,
    "Pharma trial results": 0.05,
    "Consumer spending decline": -0.22,
    "Cybersecurity breach cycle": -0.10,
    "Tariff escalation talks": -0.12,
    "Labor market surprise": 0.03,
    "Semiconductor export controls": -0.16,
}


def sentiment_label(score: float) -> str:
    if score <= -0.6:
        return "Very Bearish"
    if score <= -0.2:
        return "Bearish"
    if score < 0.2:
        return "Neutral"
    if score < 0.6:
        return "Bullish"
    return "Very Bullish"


def generate_articles(rng: np.random.Generator, embedding_dim: int = 24) -> pd.DataFrame:
    cluster_centers = rng.normal(0, 1.0, size=(5, embedding_dim))
    rows: list[dict] = []
    article_id = 1
    for event in EVENTS:
        event_drift = EVENT_SENTIMENT_DRIFT.get(event, 0.0)
        event_bias = rng.normal(event_drift, 0.16)
        for ticker in TICKERS:
            ticker_bias = rng.normal(0.0, 0.12)
            n_articles = int(rng.integers(12, 25))
            for _ in range(n_articles):
                # Mixture creates a realistic long-tail for similarity values.
                if rng.random() < 0.62:
                    similarity = float(rng.beta(7.0, 2.3))
                else:
                    similarity = float(rng.beta(1.1, 6.8))

                score = float(np.clip(rng.normal(event_bias + ticker_bias, 0.35), -1.0, 1.0))
                label = sentiment_label(score)
                cluster_id = int(rng.integers(0, 5))

                # Convert similarity [0,1] to centered semantic signal [-1,1].
                # Added noise keeps disagreement cases with sentiment.
                semantic_centered = (2.0 * similarity) - 1.0
                semantic_score = float(
                    np.clip(semantic_centered + rng.normal(0.0, 0.18), -1.0, 1.0)
                )
                sentiment_score = float(np.clip(score + rng.normal(0.0, 0.08), -1.0, 1.0))

                embedding = cluster_centers[cluster_id] + rng.normal(0, 0.45, size=embedding_dim)

                row = {
                    "article_id": article_id,
                    "event": event,
                    "ticker": ticker,
                    "overall_sentiment_score": score,
                    "sentiment_label": label,
                    "similarity_score": similarity,
                    "cluster_id": cluster_id,
                    "semantic_score_article": semantic_score,
                    "sentiment_score_article": sentiment_score,
                }
                for d in range(embedding_dim):
                    row[f"embedding_{d}"] = float(embedding[d])
                rows.append(row)
                article_id += 1
    return pd.DataFrame(rows)


def aggregate_event_ticker_scores(df_articles: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df_articles.groupby(["event", "ticker"], as_index=False)
        .agg(
            semantic_score=("semantic_score_article", "mean"),
            sentiment_score=("sentiment_score_article", "mean"),
            avg_similarity=("similarity_score", "mean"),
            article_count=("article_id", "count"),
        )
        .copy()
    )
    blended = 0.5 * grouped["semantic_score"] + 0.5 * grouped["sentiment_score"]
    # Slight positive threshold avoids too many weak-UP classifications near zero.
    grouped["predicted_direction"] = np.where(blended >= 0.05, "UP", "DOWN")
    grouped["blended_score"] = blended
    return grouped


def generate_feature_importances(rng: np.random.Generator) -> pd.DataFrame:
    # Designed to be plausible and interpretable (volatility + relative strength often high).
    base = np.array([0.07, 0.11, 0.09, 0.22, 0.08, 0.10, 0.08, 0.09, 0.16], dtype=float)
    noise = rng.normal(0.0, 0.015, size=len(base))
    imp = np.clip(base + noise, 0.01, None)
    imp = imp / imp.sum()
    return pd.DataFrame({"feature": FEATURES, "importance": imp})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic analysis datasets.")
    parser.add_argument("--output-dir", default="analysis/data", help="Directory for CSV outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--embedding-dim", type=int, default=24, help="Embedding dimensions.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    articles = generate_articles(rng=rng, embedding_dim=args.embedding_dim)
    event_ticker = aggregate_event_ticker_scores(articles)
    rf_importances = generate_feature_importances(rng)

    articles.to_csv(out_dir / "articles.csv", index=False)
    event_ticker.to_csv(out_dir / "event_ticker_scores.csv", index=False)
    rf_importances.to_csv(out_dir / "rf_feature_importances.csv", index=False)

    print(f"Wrote {len(articles)} article rows to {out_dir / 'articles.csv'}")
    print(
        "Wrote "
        f"{len(event_ticker)} event+ticker rows to {out_dir / 'event_ticker_scores.csv'}"
    )
    print(
        "Wrote "
        f"{len(rf_importances)} feature rows to {out_dir / 'rf_feature_importances.csv'}"
    )


if __name__ == "__main__":
    main()

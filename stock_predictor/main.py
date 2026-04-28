"""CLI entrypoint for ML-powered stock sentiment and prediction pipeline."""

from __future__ import annotations

import argparse

from config import DEFAULT_TOP_K, HISTORICAL_DAYS
from news_fetcher import fetch_articles_for_ticker, get_api_request_count
from output_formatter import print_results
from predictor import StockPredictor
from price_fetcher import fetch_price_data
from query_relevance import (
    build_api_search_keywords,
    cluster_articles,
    extract_keywords,
    rank_articles_by_relevance,
)
from sentiment_analyzer import SentimentAnalyzer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock sentiment + prediction pipeline using MarketAux + FinBERT + sklearn."
    )
    parser.add_argument("--query", type=str, help="Natural language query")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers, e.g. NVDA,TSM,INTC")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--history-days", type=int, default=HISTORICAL_DAYS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    query = args.query or input("Enter your query: ").strip()
    tickers_input = args.tickers or input("Enter tickers (comma-separated): ").strip()
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if not query or not tickers:
        print("Query and at least one ticker are required.")
        return

    keywords = extract_keywords(query)
    print(f"\nExtracted keywords: {keywords}\n")
    api_keywords = build_api_search_keywords(keywords)
    print(f"API search keywords: {api_keywords}\n")

    analyzer = SentimentAnalyzer()
    predictor = StockPredictor()
    all_results: dict[str, dict] = {}
    combined_articles: list[dict] = []

    for ticker in tickers:
        print(f"Fetching articles for {ticker}...")
        articles = fetch_articles_for_ticker(
            ticker,
            api_keywords,
            num_pages=args.pages,
            top_k=DEFAULT_TOP_K,
        )
        print(f"  Found {len(articles)} articles")

        relevant_articles = rank_articles_by_relevance(query, articles, top_k=DEFAULT_TOP_K)
        print(f"  Top {len(relevant_articles)} relevant articles selected")

        if relevant_articles:
            print("  Running FinBERT sentiment analysis...")
            sentiment_results = analyzer.analyze_articles(relevant_articles)
            print(f"  Sentiment: {sentiment_results['composite_score']:+.3f}")
        else:
            print(f"  No relevant articles found for {ticker}, using neutral sentiment.")
            sentiment_results = analyzer.analyze_articles([])

        try:
            price_data = fetch_price_data(ticker, period_days=args.history_days)
            print(f"  Current price: ${price_data['current_price']:.2f}")
        except Exception as exc:
            print(f"  Warning: price fetch failed for {ticker}: {exc}")
            continue

        prediction = predictor.run_prediction(price_data, sentiment_results)
        if prediction.get("rf_accuracy") is not None:
            print(
                "  RF Accuracy: "
                f"{prediction['rf_accuracy']:.2f} | LR R^2: {prediction['lr_r2']:.2f}"
            )

        all_results[ticker] = {
            "articles": relevant_articles,
            "sentiment": sentiment_results,
            "price": price_data,
            "prediction": prediction,
        }
        combined_articles.extend(sentiment_results.get("articles", []))
        print()

    cluster_info = None
    if args.cluster and combined_articles:
        print("Running optional K-Means clustering...")
        cluster_count = min(max(3, len(tickers)), len(combined_articles))
        cluster_info = cluster_articles(query, combined_articles, n_clusters=cluster_count)

    print_results(
        all_results=all_results,
        query=query,
        api_requests_used=get_api_request_count(),
        cluster_info=cluster_info,
    )


if __name__ == "__main__":
    main()

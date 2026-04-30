"""Integration test suite for stock sentiment and prediction pipeline.

These tests run against live APIs and market data, so outcomes can drift with time.
They validate directional reasonableness and pipeline invariants, not exact values.
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from typing import Callable

import polars as pl


def _synthetic_price_bundle(period_days: int) -> dict:
    """Minimal flat OHLCV history so predictors run when Yahoo has no symbol."""
    n = max(40, min(period_days, 400))
    start = date(2022, 1, 3)
    dates = [start + timedelta(days=i) for i in range(n)]
    close = [100.0 + 0.01 * i for i in range(n)]
    df = pl.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1_000_000.0] * n,
        }
    )
    from price_fetcher import compute_price_features

    feats = compute_price_features(df)
    return {"ticker": "SYNTH", "history_df": df, **feats}


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.checks: list[tuple[str, str]] = []

    def check(self, condition: bool, description: str) -> None:
        status = "PASS" if condition else "FAIL"
        self.checks.append((status, description))
        print(f"  [{status}] {description}")

    def summary(self) -> tuple[int, int]:
        passed = sum(1 for s, _ in self.checks if s == "PASS")
        return passed, len(self.checks)


def run_pipeline_for_test(
    query: str,
    tickers: list[str],
    top_k: int = 5,
    pages: int = 3,
    history_days: int = 180,
):
    """Run the full pipeline and return structured outputs."""
    from news_fetcher import fetch_articles_for_ticker
    from predictor import StockPredictor
    from price_fetcher import fetch_price_data
    from query_relevance import (
        build_api_search_keywords,
        cluster_articles,
        extract_keywords,
        rank_articles_by_relevance,
    )
    from sentiment_analyzer import SentimentAnalyzer

    keywords = extract_keywords(query)
    api_keywords = build_api_search_keywords(keywords)
    analyzer = SentimentAnalyzer()

    all_results: dict[str, dict] = {}
    all_articles_flat: list[dict] = []

    for ticker in tickers:
        articles = fetch_articles_for_ticker(
            ticker=ticker, search_keywords=api_keywords, num_pages=pages, top_k=top_k
        )
        relevant = rank_articles_by_relevance(query, articles, top_k=top_k) if articles else []

        if relevant:
            sentiment_results = analyzer.analyze_articles(relevant)
        else:
            sentiment_results = {
                "articles": [],
                "finbert_avg": 0.0,
                "marketaux_avg": 0.0,
                "composite_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "max_sentiment": 0.0,
                "min_sentiment": 0.0,
                "sentiment_std": 0.0,
                "strongest_positive_headline": "",
                "strongest_negative_headline": "",
            }

        try:
            price_data = fetch_price_data(ticker, period_days=history_days)
        except ValueError:
            price_data = _synthetic_price_bundle(period_days=history_days)

        predictor = StockPredictor()
        prediction = predictor.run_prediction(price_data, sentiment_results)

        all_results[ticker] = {
            "articles": relevant,
            "sentiment": sentiment_results,
            "price": price_data,
            "prediction": prediction,
            "keywords": keywords,
            "api_keywords": api_keywords,
        }
        all_articles_flat.extend(relevant)
        time.sleep(0.5)

    cluster_info = None
    if len(all_articles_flat) >= 3:
        try:
            cluster_info = cluster_articles(
                query=query,
                all_articles=all_articles_flat,
                n_clusters=min(3, max(1, len(tickers))),
            )
        except Exception:
            cluster_info = None

    return all_results, cluster_info


def test_1_ai_semiconductor_bull():
    test = TestResult("TEST 1: AI Semiconductor Bull Case")
    print(f"\n{'=' * 60}\n  {test.name}\n{'=' * 60}")

    results, clusters = run_pipeline_for_test(
        query="How will AI chip demand affect semiconductor stocks?",
        tickers=["NVDA", "MSFT", "INTC"],
        top_k=5,
        pages=3,
    )

    for ticker in ["NVDA", "MSFT", "INTC"]:
        r = results[ticker]
        test.check(len(r["articles"]) > 0, f"{ticker}: fetched >=1 article ({len(r['articles'])})")
        test.check(r["sentiment"]["composite_score"] > -0.55, f"{ticker}: sentiment not strongly negative ({r['sentiment']['composite_score']:.3f})")
        test.check(r["prediction"]["direction"] in ["UP", "DOWN"], f"{ticker}: direction valid ({r['prediction']['direction']})")
        test.check(0 <= r["prediction"]["magnitude_pct"] < 50, f"{ticker}: magnitude reasonable ({r['prediction']['magnitude_pct']:.2f}%)")
        test.check(r["prediction"].get("sentiment_weight_used", 0) >= 0.15, f"{ticker}: sentiment has non-trivial weight ({r['prediction'].get('sentiment_weight_used', 0):.2f})")

    test.check(clusters is None or clusters.get("enabled", False), "Clustering output valid when available")
    return test, results


def test_2_oil_geopolitical():
    test = TestResult("TEST 2: Oil Geopolitical")
    print(f"\n{'=' * 60}\n  {test.name}\n{'=' * 60}")

    results, _ = run_pipeline_for_test(
        query="How will the Iran conflict and oil prices impact energy stocks?",
        tickers=["XOM", "CVX"],
        top_k=5,
        pages=3,
    )

    for ticker in ["XOM", "CVX"]:
        r = results[ticker]
        test.check(len(r["articles"]) > 0, f"{ticker}: fetched >=1 article ({len(r['articles'])})")
        test.check(r["price"]["current_price"] > 0, f"{ticker}: valid current price ({r['price']['current_price']:.2f})")
        test.check(0 <= r["prediction"]["magnitude_pct"] < 30, f"{ticker}: magnitude reasonable ({r['prediction']['magnitude_pct']:.2f}%)")
        test.check(r["price"].get("volatility_20d", 0) >= 0, f"{ticker}: volatility feature available ({r['price'].get('volatility_20d', 0):.4f})")

    return test, results


def test_3_tesla_mixed():
    test = TestResult("TEST 3: Tesla Mixed Sentiment")
    print(f"\n{'=' * 60}\n  {test.name}\n{'=' * 60}")

    results, _ = run_pipeline_for_test(
        query="Tesla earnings delivery miss EV competition outlook",
        tickers=["TSLA"],
        top_k=5,
        pages=3,
    )

    r = results["TSLA"]
    test.check(len(r["articles"]) > 0, f"TSLA: fetched >=1 article ({len(r['articles'])})")
    if r["articles"]:
        pos = r["sentiment"]["positive_count"]
        neg = r["sentiment"]["negative_count"]
        neu = r["sentiment"]["neutral_count"]
        has_variety = (pos > 0 and neg > 0) or (pos > 0 and neu > 0) or (neg > 0 and neu > 0)
        strong_tone = abs(r["sentiment"]["composite_score"]) >= 0.35
        test.check(
            has_variety or strong_tone,
            f"TSLA: sentiment variety or strong net tone (pos={pos}, neg={neg}, neu={neu}, comp={r['sentiment']['composite_score']:.3f})",
        )
        test.check(-1.0 <= r["sentiment"]["composite_score"] <= 1.0, f"TSLA: composite in range ({r['sentiment']['composite_score']:.3f})")
    test.check(r["prediction"]["direction"] in ["UP", "DOWN"], f"TSLA: direction valid ({r['prediction']['direction']})")

    return test, results


def test_4_banking_rates():
    test = TestResult("TEST 4: Banking and Rates")
    print(f"\n{'=' * 60}\n  {test.name}\n{'=' * 60}")

    results, _ = run_pipeline_for_test(
        query="How do interest rates and Fed policy affect bank earnings?",
        tickers=["JPM", "GS"],
        top_k=5,
        pages=3,
    )

    for ticker in ["JPM", "GS"]:
        r = results[ticker]
        test.check(len(r["articles"]) > 0, f"{ticker}: fetched >=1 article ({len(r['articles'])})")
        test.check(r["sentiment"]["composite_score"] >= -0.5, f"{ticker}: sentiment not extreme negative ({r['sentiment']['composite_score']:.3f})")
        test.check(0 <= r["prediction"]["magnitude_pct"] < 20, f"{ticker}: magnitude reasonable ({r['prediction']['magnitude_pct']:.2f}%)")

    return test, results


def test_5_edge_case_low_coverage():
    test = TestResult("TEST 5: Edge Case Low Coverage")
    print(f"\n{'=' * 60}\n  {test.name}\n{'=' * 60}")

    try:
        results, _ = run_pipeline_for_test(
            query="Small cap industrial automation growth",
            tickers=["ROCKU"],
            top_k=5,
            pages=1,
        )
        r = results["ROCKU"]
        test.check(True, "Pipeline completed without crash")
        if len(r["articles"]) == 0:
            test.check(r["sentiment"]["composite_score"] == 0.0, "No articles => neutral fallback sentiment")
        test.check(r["prediction"]["direction"] in ["UP", "DOWN"], f"Direction produced ({r['prediction']['direction']})")
        return test, results
    except Exception as exc:
        test.check(False, f"Pipeline crashed on low coverage ticker: {type(exc).__name__}: {exc}")
        return test, {}


def test_cross_cutting(all_test_results: dict[str, dict[str, dict]]) -> TestResult:
    test = TestResult("CROSS-CUTTING: Pipeline Invariants")
    print(f"\n{'=' * 60}\n  {test.name}\n{'=' * 60}")

    for test_name, results in all_test_results.items():
        for ticker, r in results.items():
            comp = r["sentiment"]["composite_score"]
            mag = r["prediction"]["magnitude_pct"]
            conf = r["prediction"]["confidence"]
            sw = r["prediction"].get("sentiment_weight_used", 0.0)
            mw = r["prediction"].get("model_weight_used", 0.0)
            article_count = len(r["articles"])
            sentiment_count = (
                r["sentiment"]["positive_count"]
                + r["sentiment"]["negative_count"]
                + r["sentiment"]["neutral_count"]
            )

            test.check(-1.0 <= comp <= 1.0, f"[{test_name}] {ticker}: composite in [-1,1] ({comp:.3f})")
            test.check(mag >= 0, f"[{test_name}] {ticker}: magnitude non-negative ({mag:.2f}%)")
            test.check(0 <= conf <= 1.0, f"[{test_name}] {ticker}: confidence in [0,1] ({conf:.3f})")
            test.check(r["prediction"]["direction"] in ["UP", "DOWN"], f"[{test_name}] {ticker}: direction valid")
            if sw > 0 or mw > 0:
                test.check(0.95 <= sw + mw <= 1.05, f"[{test_name}] {ticker}: blend weights sum ~1 ({sw + mw:.2f})")
            test.check(article_count == sentiment_count, f"[{test_name}] {ticker}: article count matches sentiment label count ({article_count})")

    return test


if __name__ == "__main__":
    print("=" * 60)
    print("  STOCK PREDICTION PIPELINE - TEST SUITE")
    print("=" * 60)
    print("\nWARNING: These tests use local news + Yahoo Finance; outcomes can drift over time.\n")

    selected = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 3, 4, 5]

    test_map: dict[int, tuple[str, Callable]] = {
        1: ("AI Semiconductor", test_1_ai_semiconductor_bull),
        2: ("Oil Geopolitical", test_2_oil_geopolitical),
        3: ("Tesla Mixed", test_3_tesla_mixed),
        4: ("Banking Rates", test_4_banking_rates),
        5: ("Edge Case", test_5_edge_case_low_coverage),
    }

    all_tests: list[TestResult] = []
    all_results_for_crosscut: dict[str, dict[str, dict]] = {}

    for num in selected:
        if num not in test_map:
            continue
        name, func = test_map[num]
        try:
            t, results = func()
            all_tests.append(t)
            all_results_for_crosscut[name] = results
        except Exception as exc:
            print(f"\n[CRASH] {name} threw an exception: {exc}")

    if all_results_for_crosscut:
        cross = test_cross_cutting(all_results_for_crosscut)
        all_tests.append(cross)

    print(f"\n{'=' * 60}\n  FINAL SUMMARY\n{'=' * 60}")
    total_passed = 0
    total_checks = 0
    for t in all_tests:
        p, total = t.summary()
        total_passed += p
        total_checks += total
        status = "ALL PASSED" if p == total else f"{total - p} FAILED"
        print(f"  {t.name}: {p}/{total} ({status})")

    print(f"\n  OVERALL: {total_passed}/{total_checks} checks passed")
    if total_passed == total_checks:
        print("  Result: ALL TESTS PASSED")
    else:
        print(f"  Result: {total_checks - total_passed} CHECKS FAILED")
    print("=" * 60)
    sys.exit(0 if total_passed == total_checks else 1)

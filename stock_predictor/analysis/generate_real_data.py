"""Run the live pipeline and write analysis CSVs (same layout as ``generate_test_data``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_TOP_K, HISTORICAL_DAYS
from feature_preprocessing import apply_column_mask, winsorize_apply
from news_fetcher import fetch_articles_for_ticker, reset_api_request_count
from predictor import StockPredictor
from price_fetcher import fetch_price_data
from query_relevance import build_api_search_keywords, extract_keywords, rank_articles_by_relevance
from sentiment_analyzer import SentimentAnalyzer

DEFAULT_EVENTS = [
    "How will AI chip demand affect semiconductor stocks?",
    "How will rates policy affect bank earnings?",
    "How will oil volatility affect energy stocks?",
    "How will cloud AI spending affect big tech?",
]

DEFAULT_TICKERS = ["NVDA", "TSM", "AAPL", "MSFT", "JPM", "GS", "XOM", "CVX"]

PREDICTOR_FEATURES = [
    "finbert_avg",
    "marketaux_avg",
    "composite_score",
    "positive_ratio",
    "negative_ratio",
    "max_sentiment",
    "min_sentiment",
    "sentiment_std",
    "returns_5d",
    "returns_10d",
    "returns_20d",
    "volatility_20d",
    "volume_ratio",
    "rsi_14",
    "sma_cross",
]


def evaluate_baselines_after_run(
    predictor: StockPredictor,
    price_df: pl.DataFrame,
    sentiment: dict[str, Any],
) -> list[dict[str, Any]]:
    X, y_class, _ = predictor.build_training_data(price_df, sentiment)
    if len(X) < 10 or predictor._rf_best is None:
        return []
    if predictor._winsor_low is None or predictor._winsor_high is None or predictor._feature_mask is None:
        return []

    split_idx = max(1, int(len(X) * 0.7))
    split_idx = min(split_idx, len(X) - 1)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y_class[:split_idx], y_class[split_idx:]
    if len(X_test) == 0:
        return []

    X_train = apply_column_mask(
        winsorize_apply(X_train_raw, predictor._winsor_low, predictor._winsor_high),
        predictor._feature_mask,
    )
    X_test = apply_column_mask(
        winsorize_apply(X_test_raw, predictor._winsor_low, predictor._winsor_high),
        predictor._feature_mask,
    )

    def _one_row(model_name: str, model: Any, needs_fit: bool) -> dict[str, Any]:
        if needs_fit:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0))
        auc = float("nan")
        if len(np.unique(y_test)) > 1 and hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)[:, 1]
                auc = float(roc_auc_score(y_test, probs))
            except Exception:
                auc = float("nan")
        return {
            "model": model_name,
            "accuracy": acc,
            "f1": f1,
            "roc_auc": auc,
            "test_samples": int(len(X_test)),
        }

    return [
        _one_row("RandomForest (current)", predictor._rf_best, needs_fit=False),
        _one_row(
            "Dummy Uniform (baseline)",
            DummyClassifier(strategy="uniform", random_state=42),
            needs_fit=True,
        ),
        _one_row(
            "LogisticRegression",
            SkPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
                ]
            ),
            needs_fit=True,
        ),
    ]


def _append_importance_rows(
    predictor: StockPredictor,
    event: str,
    ticker: str,
    prediction: dict[str, Any],
    importance_rows: list[dict[str, Any]],
) -> None:
    if prediction.get("rf_accuracy") is None or predictor._rf_best is None or predictor._feature_mask is None:
        return
    try:
        clf = predictor._rf_best.named_steps["clf"]
        imp = np.asarray(getattr(clf, "feature_importances_", np.array([])), dtype=float)
        mask = predictor._feature_mask.astype(bool)
        if imp.size == 0 or mask.shape[0] != len(PREDICTOR_FEATURES) or imp.size != int(mask.sum()):
            return
        names = [n for n, keep in zip(PREDICTOR_FEATURES, mask) if keep]
        if len(names) != len(imp):
            return
        for name, value in zip(names, imp):
            importance_rows.append(
                {
                    "event": event,
                    "ticker": ticker,
                    "feature": name,
                    "importance": float(value),
                }
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        return


def sentiment_bucket(score: float) -> str:
    if score <= -0.6:
        return "Very Bearish"
    if score <= -0.2:
        return "Bearish"
    if score < 0.2:
        return "Neutral"
    if score < 0.6:
        return "Bullish"
    return "Very Bullish"


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real-data analysis CSVs via the pipeline.")
    parser.add_argument("--output-dir", default="analysis/data_real")
    parser.add_argument("--events", default="|".join(DEFAULT_EVENTS), help="Pipe-separated events.")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers.")
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--history-days", type=int, default=HISTORICAL_DAYS)
    parser.add_argument("--n-clusters", type=int, default=5, help="KMeans clusters for TF-IDF article vectors.")
    parser.add_argument("--embedding-dim", type=int, default=64, help="TF-IDF max features for embeddings.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = [e.strip() for e in args.events.split("|") if e.strip()]
    tickers = [t.upper() for t in parse_csv(args.tickers)]

    reset_api_request_count()
    analyzer = SentimentAnalyzer()

    article_rows: list[dict[str, Any]] = []
    event_ticker_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    model_metric_rows: list[dict[str, Any]] = []

    article_id = 1
    for event in events:
        keywords = extract_keywords(event)
        api_keywords = build_api_search_keywords(keywords)

        for ticker in tickers:
            articles = fetch_articles_for_ticker(
                ticker=ticker,
                search_keywords=api_keywords,
                num_pages=args.pages,
                top_k=args.top_k,
            )
            ranked = rank_articles_by_relevance(event, articles, top_k=args.top_k)
            sentiment = analyzer.analyze_articles(ranked)

            semantic_score = (
                float(np.mean([float(a.get("relevance_score", 0.0) or 0.0) for a in ranked]))
                if ranked
                else 0.0
            )
            sentiment_score = float(sentiment.get("composite_score", 0.0))

            predicted_direction = "DOWN"
            try:
                predictor = StockPredictor()
                price_data = fetch_price_data(ticker, period_days=args.history_days)
                prediction = predictor.run_prediction(price_data, sentiment)
                predicted_direction = str(prediction.get("direction", "DOWN"))
                for row in evaluate_baselines_after_run(
                    predictor=predictor,
                    price_df=price_data["history_df"],
                    sentiment=sentiment,
                ):
                    model_metric_rows.append({"event": event, "ticker": ticker, **row})
                _append_importance_rows(predictor, event, ticker, prediction, importance_rows)
            except Exception as exc:
                print(f"Warning: prediction failed for {ticker} on '{event}': {exc}")

            event_ticker_rows.append(
                {
                    "event": event,
                    "ticker": ticker,
                    "semantic_score": semantic_score,
                    "sentiment_score": sentiment_score,
                    "predicted_direction": predicted_direction,
                    "article_count": len(sentiment.get("articles", [])),
                }
            )

            for a in sentiment.get("articles", []):
                finbert_score = float(a.get("finbert", {}).get("score", 0.0))
                article_rows.append(
                    {
                        "article_id": article_id,
                        "event": event,
                        "ticker": ticker,
                        "overall_sentiment_score": finbert_score,
                        "sentiment_label": sentiment_bucket(finbert_score),
                        "similarity_score": float(a.get("relevance_score", 0.0) or 0.0),
                        "cluster_id": -1,
                        "title": a.get("title", ""),
                        "description": a.get("description", ""),
                        "snippet": a.get("snippet", ""),
                    }
                )
                article_id += 1

    df_articles = pl.DataFrame(article_rows) if article_rows else pl.DataFrame()
    df_event_ticker = pl.DataFrame(event_ticker_rows) if event_ticker_rows else pl.DataFrame()
    df_importance = pl.DataFrame(importance_rows) if importance_rows else pl.DataFrame()
    df_model_metrics = pl.DataFrame(model_metric_rows) if model_metric_rows else pl.DataFrame()

    if not df_articles.is_empty():
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        docs = (
            pl.col("title").fill_null("")
            + pl.lit(" ")
            + pl.col("description").fill_null("")
            + pl.lit(" ")
            + pl.col("snippet").fill_null("")
        )
        text_series = df_articles.with_columns(docs.alias("_doc"))["_doc"].to_list()
        tfidf = TfidfVectorizer(stop_words="english", max_features=args.embedding_dim)
        matrix = tfidf.fit_transform(text_series).toarray()

        cluster_count = min(max(1, args.n_clusters), len(df_articles))
        if cluster_count >= 2 and matrix.shape[0] >= cluster_count:
            km = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
            cluster_ids = km.fit_predict(matrix)
            df_articles = df_articles.with_columns(pl.Series("cluster_id", cluster_ids))
        else:
            df_articles = df_articles.with_columns(pl.lit(0).alias("cluster_id"))

        for i in range(matrix.shape[1]):
            df_articles = df_articles.with_columns(pl.Series(f"embedding_{i}", matrix[:, i]))

    if not df_importance.is_empty():
        df_importance = df_importance.group_by("feature").agg(pl.col("importance").mean().alias("importance"))
        total = float(df_importance["importance"].sum())
        if total > 0:
            df_importance = df_importance.with_columns((pl.col("importance") / total).alias("importance"))

    df_articles.write_csv(output_dir / "articles.csv")
    df_event_ticker.write_csv(output_dir / "event_ticker_scores.csv")
    df_importance.write_csv(output_dir / "rf_feature_importances.csv")
    df_model_metrics.write_csv(output_dir / "model_metrics.csv")

    print(f"Wrote {df_articles.height} rows -> {output_dir / 'articles.csv'}")
    print(f"Wrote {df_event_ticker.height} rows -> {output_dir / 'event_ticker_scores.csv'}")
    print(f"Wrote {df_importance.height} rows -> {output_dir / 'rf_feature_importances.csv'}")
    print(f"Wrote {df_model_metrics.height} rows -> {output_dir / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()

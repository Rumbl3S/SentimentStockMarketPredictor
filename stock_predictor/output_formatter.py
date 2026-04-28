"""Console formatting for stock prediction results."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def _score_label(score: float) -> str:
    if score > 0.1:
        return "Positive"
    if score < -0.1:
        return "Negative"
    return "Neutral"


def _safe_date(iso_string: str) -> str:
    try:
        return datetime.fromisoformat(iso_string.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except Exception:
        return iso_string[:10] if iso_string else "N/A"


def clean_highlight(text: str) -> str:
    """Clean up MarketAux highlight text for display."""
    cleaned = re.sub(r"</?em>", "", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\+\d+ characters?\]", "...", cleaned)
    cleaned = re.sub(r"<[^>]*$", "", cleaned)
    return cleaned.strip()


def print_results(
    all_results: dict[str, dict[str, Any]],
    query: str,
    api_requests_used: int,
    cluster_info: dict[str, Any] | None = None,
) -> None:
    print("\n" + "═" * 62)
    print(f'QUERY: "{query}"')
    print("═" * 62)

    for ticker, result in all_results.items():
        sentiment = result["sentiment"]
        prediction = result["prediction"]
        price = result["price"]
        articles = sentiment.get("articles", [])

        direction = prediction["direction"]
        pct = prediction["magnitude_pct"]
        conf = prediction["confidence"] * 100
        signal_text = prediction.get("signal") or (
            "Low Confidence - Mixed Signals"
            if prediction.get("mixed_signals")
            else "Sentiment and price momentum AGREE"
        )
        signed_pct = pct if direction == "UP" else -pct
        move_text = f"{signed_pct:+.2f}%"

        print("\n" + "═" * 62)
        print(f"TICKER: {ticker}")
        print("═" * 62)
        print(f"\nPREDICTION: {direction} {move_text}  (Confidence: {conf:.1f}%)")
        print(f"   Signal: {signal_text}")
        model_weight = float(prediction.get("model_weight_used", 0.0)) * 100
        sentiment_weight = float(prediction.get("sentiment_weight_used", 0.0)) * 100
        rf_acc = prediction.get("rf_accuracy")
        rf_label = f"{rf_acc:.2f} RF acc" if isinstance(rf_acc, float) else "RF acc n/a"
        print(
            "   Weights: "
            f"{model_weight:.0f}% price model ({rf_label}) / "
            f"{sentiment_weight:.0f}% sentiment"
        )
        print("\nSENTIMENT SUMMARY:")
        print(
            f"   FinBERT Average:      {sentiment['finbert_avg']:+.2f} "
            f"({_score_label(sentiment['finbert_avg'])})"
        )
        print(
            f"   MarketAux Average:    {sentiment['marketaux_avg']:+.2f} "
            f"({_score_label(sentiment['marketaux_avg'])})"
        )
        print(f"   Composite Score:      {sentiment['composite_score']:+.2f}")
        print(
            f"   Articles Analyzed:    {len(articles)} "
            f"({sentiment['positive_count']} positive, "
            f"{sentiment['negative_count']} negative, "
            f"{sentiment['neutral_count']} neutral)"
        )
        print("\nPRICE FEATURES:")
        print(f"   Current Price:   ${price['current_price']:.2f}")
        print(f"   5-Day Return:    {price['returns_5d']:+.2f}%")
        print(f"   20-Day Return:   {price['returns_20d']:+.2f}%")
        print(f"   RSI(14):         {price['rsi_14']:.2f}")

        if prediction.get("rf_accuracy") is not None:
            print(
                f"   RF Accuracy:     {prediction['rf_accuracy']:.2f} | "
                f"LR R^2: {prediction['lr_r2']:.2f}"
            )

        print("\nCITED ARTICLES:")
        if not articles:
            print("   No relevant articles available.")
        for idx, article in enumerate(articles, start=1):
            finbert_score = article.get("finbert", {}).get("score", 0.0)
            title = article.get("title", "Untitled")
            source = article.get("source", "unknown")
            date_str = _safe_date(article.get("published_at", ""))
            top_highlight = ""
            if article.get("highlights"):
                top_highlight = clean_highlight(article["highlights"][0].get("highlight", ""))
            print(
                f'   {idx}. [{finbert_score:+.2f}] "{title}" '
                f"- {source} ({date_str})"
            )
            if top_highlight:
                print(f'      Highlight: "{top_highlight}"')
            print(f"      URL: {article.get('url', '')}")

    if cluster_info and cluster_info.get("enabled"):
        print("\n" + "─" * 62)
        print("OPTIONAL CLUSTERING SUMMARY")
        clusters = cluster_info.get("clusters", {})
        for cid in sorted(clusters):
            info = clusters[cid]
            top_ticker = info.get("top_tickers", [("mixed", 0)])[0]
            interpretation = (
                f"{top_ticker[0]}-focused coverage ({info['size']} articles)"
                if top_ticker[1] > 0
                else f"Mixed coverage ({info['size']} articles)"
            )
            print(
                f"  Cluster {cid}: {interpretation} | "
                f"tickers={', '.join([f'{t}:{c}' for t, c in info.get('top_tickers', [])])} | "
                f"top_terms={', '.join(info['top_terms'][:5])}"
            )
        query_cluster = cluster_info.get("query_cluster")
        query_terms = ""
        if query_cluster in clusters:
            query_terms = ", ".join(clusters[query_cluster].get("top_terms", [])[:5])
        print(
            f"  Your query is most related to Cluster {query_cluster}: "
            f"{query_terms or 'no terms available'}"
        )

    print("\n" + "─" * 62)
    print(f"API Requests Used: {api_requests_used}/100 daily limit")
    print("This is a class project and not financial advice.")

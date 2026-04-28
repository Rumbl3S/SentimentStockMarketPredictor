"""FinBERT sentiment analysis and per-ticker aggregates."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]


class SentimentAnalyzer:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()

    def analyze_text(self, text: str) -> dict[str, Any]:
        """Return normalized FinBERT sentiment outputs."""
        cleaned = (text or "").strip()
        if not cleaned:
            return {
                "score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "label": "neutral",
            }

        encoded = self.tokenizer(
            cleaned,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        probs_map = dict(zip(LABELS, probs))
        score = float(probs_map["positive"] - probs_map["negative"])
        label = max(probs_map, key=probs_map.get)
        return {
            "score": score,
            "positive": float(probs_map["positive"]),
            "negative": float(probs_map["negative"]),
            "neutral": float(probs_map["neutral"]),
            "label": label,
        }

    def analyze_articles(self, articles: list[dict[str, Any]]) -> dict[str, Any]:
        """Run FinBERT on each article and return aggregate stats."""
        enriched: list[dict[str, Any]] = []
        finbert_scores: list[float] = []
        marketaux_scores: list[float] = []
        relevance_weights: list[float] = []
        marketaux_weights: list[float] = []

        for article in articles:
            highlight_text = " ".join(
                str(h.get("highlight", "")) for h in article.get("highlights", [])
            )
            text = ". ".join(
                [
                    article.get("title", ""),
                    article.get("description", ""),
                    article.get("snippet", ""),
                    highlight_text,
                ]
            )
            finbert = self.analyze_text(text)
            item = dict(article)
            item["finbert"] = finbert
            enriched.append(item)
            finbert_scores.append(finbert["score"])
            relevance = float(article.get("relevance_score", 0.0) or 0.0)
            relevance_weights.append(max(0.0, relevance))

            raw_marketaux = article.get("entity_sentiment_score")
            if raw_marketaux is not None:
                marketaux_scores.append(float(raw_marketaux))
                marketaux_weights.append(max(0.0, relevance))

        if finbert_scores:
            if sum(relevance_weights) > 0:
                finbert_avg = float(np.average(finbert_scores, weights=relevance_weights))
            else:
                finbert_avg = float(np.mean(finbert_scores))
        else:
            finbert_avg = 0.0

        if marketaux_scores:
            if sum(marketaux_weights) > 0:
                marketaux_avg = float(np.average(marketaux_scores, weights=marketaux_weights))
            else:
                marketaux_avg = float(np.mean(marketaux_scores))
        else:
            marketaux_avg = 0.0
        composite_score = 0.6 * finbert_avg + 0.4 * marketaux_avg

        labels = [a["finbert"]["label"] for a in enriched]
        pos_count = sum(1 for l in labels if l == "positive")
        neg_count = sum(1 for l in labels if l == "negative")
        neutral_count = sum(1 for l in labels if l == "neutral")

        strongest_pos = max(enriched, key=lambda a: a["finbert"]["score"], default=None)
        strongest_neg = min(enriched, key=lambda a: a["finbert"]["score"], default=None)

        return {
            "articles": enriched,
            "finbert_avg": finbert_avg,
            "marketaux_avg": marketaux_avg,
            "composite_score": composite_score,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neutral_count,
            "positive_ratio": (pos_count / len(enriched)) if enriched else 0.0,
            "negative_ratio": (neg_count / len(enriched)) if enriched else 0.0,
            "max_sentiment": max(finbert_scores) if finbert_scores else 0.0,
            "min_sentiment": min(finbert_scores) if finbert_scores else 0.0,
            "sentiment_std": float(np.std(finbert_scores)) if finbert_scores else 0.0,
            "strongest_positive_headline": (
                strongest_pos.get("title", "") if strongest_pos else ""
            ),
            "strongest_negative_headline": (
                strongest_neg.get("title", "") if strongest_neg else ""
            ),
        }

"""Build a large local training dataset from RSS news + Yahoo prices.

This script is designed for class data requirements:
- web API / web scrape collection
- local persisted datasets
- >=50k cleaned rows in final merged dataset
- 7-10+ feature columns
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import yfinance as yf


DEFAULT_SEED_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "ABBV", "PFE", "KO",
    "AVGO", "COST", "PEP", "MRK", "BAC", "WMT", "CSCO", "ADBE", "NFLX", "TMO",
    "ACN", "CRM", "MCD", "DHR", "DIS", "ABT", "VZ", "INTC", "CMCSA", "WFC",
    "AMD", "TXN", "LIN", "NEE", "BMY", "PM", "UNP", "HON", "QCOM", "LOW",
    "SBUX", "ORCL", "AMGN", "UPS", "GS", "MS", "BLK", "CAT", "BA", "RTX",
    "IBM", "SPGI", "GE", "NOW", "AMAT", "DE", "PLD", "GILD", "MDT", "BKNG",
    "ADP", "LMT", "CVS", "TJX", "MO", "AXP", "C", "SYK", "ISRG", "T",
    "MMC", "SCHW", "MDLZ", "ELV", "CI", "NKE", "REGN", "VRTX", "INTU", "MU",
]

TICKER_ALIASES = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "GOOGL": ["alphabet", "google"],
    "AMZN": ["amazon"],
    "META": ["meta", "facebook"],
    "NVDA": ["nvidia"],
    "TSLA": ["tesla"],
    "BRK-B": ["berkshire hathaway", "berkshire"],
    "BA": ["boeing"],
    "JPM": ["jpmorgan", "jp morgan"],
    "GS": ["goldman sachs"],
    "XOM": ["exxon", "exxonmobil"],
    "CVX": ["chevron"],
}

POSITIVE_WORDS = {"beat", "surge", "rally", "growth", "gain", "upgrade", "strong", "profit"}
NEGATIVE_WORDS = {"miss", "drop", "fall", "downgrade", "lawsuit", "crash", "loss", "decline"}


@dataclass
class BuildStats:
    raw_news_rows: int = 0
    clean_news_rows: int = 0
    raw_price_rows: int = 0
    feature_price_rows: int = 0
    final_rows: int = 0


def _normalize_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    return t.replace(".", "-")


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        norm = _normalize_ticker(item)
        if not norm or norm in seen:
            continue
        deduped.append(norm)
        seen.add(norm)
    return deduped


def load_tickers_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    rows = path.read_text(encoding="utf-8").splitlines()
    return _dedupe_keep_order(rows)


def fetch_sp500_tickers() -> list[str]:
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return _dedupe_keep_order(table["Symbol"].astype(str).tolist())
    except Exception:
        return []


def fetch_nasdaq100_tickers() -> list[str]:
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
        column = "Ticker" if "Ticker" in table.columns else table.columns[1]
        return _dedupe_keep_order(table[column].astype(str).tolist())
    except Exception:
        return []


def resolve_ticker_universe(
    ticker_source: str,
    manual_tickers: list[str],
    tickers_file: Path | None,
    min_count: int = 50,
) -> list[str]:
    source_tickers: list[str] = []
    if ticker_source == "sp500":
        source_tickers = fetch_sp500_tickers()
    elif ticker_source == "nasdaq100":
        source_tickers = fetch_nasdaq100_tickers()
    elif ticker_source == "all":
        source_tickers = fetch_sp500_tickers() + fetch_nasdaq100_tickers() + DEFAULT_SEED_TICKERS
    else:
        source_tickers = DEFAULT_SEED_TICKERS

    file_tickers = load_tickers_from_file(tickers_file) if tickers_file else []
    combined = _dedupe_keep_order(source_tickers + file_tickers + manual_tickers)

    # Safety fallback: never run tiny universes unless explicitly requested.
    if len(combined) < min_count:
        combined = _dedupe_keep_order(combined + DEFAULT_SEED_TICKERS)
    return combined


def ensure_dirs(root: Path) -> dict[str, Path]:
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    final = root / "data" / "final"
    for d in [raw, processed, final]:
        d.mkdir(parents=True, exist_ok=True)
    return {"raw": raw, "processed": processed, "final": final}


def parse_pub_date(raw: str) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def rss_for_ticker(ticker: str, lookback_days: int, timeout: int = 20) -> list[dict]:
    aliases = TICKER_ALIASES.get(ticker, [])
    primary_name = aliases[0] if aliases else ticker
    query_templates = [
        f"{ticker} stock when:{lookback_days}d",
        f"{primary_name} stock when:{lookback_days}d",
        f"{primary_name} earnings guidance when:{lookback_days}d",
        f"{primary_name} lawsuit investigation risk when:{lookback_days}d",
        f"{primary_name} analyst target outlook when:{lookback_days}d",
    ]
    rows: list[dict] = []
    for q in query_templates:
        query = quote_plus(q)
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException:
            continue

        root = ET.fromstring(response.content)
        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_raw = (item.findtext("pubDate") or "").strip()
            source = (item.findtext("source") or "").strip()
            dt = parse_pub_date(pub_raw)
            description = (item.findtext("description") or "").strip()
            rows.append(
                {
                    "ticker_seed": ticker,
                    "query_template": q,
                    "title": title,
                    "url": link,
                    "source": source,
                    "published_at": dt.isoformat() if dt else "",
                    "description": description,
                }
            )
    return rows


def collect_rss_news(tickers: Iterable[str], lookback_days: int) -> pd.DataFrame:
    rows: list[dict] = []
    for ticker in tickers:
        rows.extend(rss_for_ticker(ticker, lookback_days=lookback_days))
    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker_seed",
                "query_template",
                "title",
                "url",
                "source",
                "published_at",
                "description",
            ]
        )
    news = pd.DataFrame(rows)
    news = news.drop_duplicates(subset=["url", "title"]).copy()
    news["date"] = pd.to_datetime(news["published_at"], errors="coerce").dt.date
    return news


def find_ticker_matches(text: str, tickers: Iterable[str]) -> list[str]:
    lowered = (text or "").lower()
    matched: list[str] = []
    for t in tickers:
        aliases = TICKER_ALIASES.get(t, [])
        symbol_re = re.compile(rf"\b{re.escape(t.lower())}\b")
        if symbol_re.search(lowered):
            matched.append(t)
            continue
        if any(a in lowered for a in aliases):
            matched.append(t)
    return matched


def pseudo_highlight(text: str, ticker: str) -> str:
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    aliases = [ticker.lower()] + TICKER_ALIASES.get(ticker, [])
    for s in sentences:
        lowered = s.lower()
        if any(a in lowered for a in aliases):
            return s[:280]
    return (sentences[0] if sentences else text)[:280]


def simple_sentiment(text: str) -> float:
    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    score = (pos - neg) / max(1, math.sqrt(len(words)))
    return float(np.clip(score, -1.0, 1.0))


def build_news_features(news_df: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if news_df.empty:
        empty_articles = pd.DataFrame(
            columns=["ticker", "date", "title", "url", "source", "highlight", "sentiment_score", "relevance_score"]
        )
        empty_daily = pd.DataFrame(
            columns=["ticker", "date", "article_count", "avg_sentiment", "sentiment_std", "positive_ratio", "negative_ratio", "avg_relevance"]
        )
        return empty_articles, empty_daily

    mapped_rows: list[dict] = []
    for _, row in news_df.iterrows():
        text_blob = f"{row.get('title', '')}. {row.get('description', '')}"
        matched = find_ticker_matches(text_blob, tickers)
        if not matched:
            matched = [row.get("ticker_seed", "")]
        for ticker in matched:
            if not ticker:
                continue
            score = simple_sentiment(text_blob)
            rel = min(1.0, 0.15 + (0.1 * len(find_ticker_matches(text_blob, [ticker]))))
            mapped_rows.append(
                {
                    "ticker": ticker,
                    "date": row.get("date"),
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "url": row.get("url", ""),
                    "source": row.get("source", ""),
                    "highlight": pseudo_highlight(text_blob, ticker),
                    "sentiment_score": score,
                    "relevance_score": rel,
                }
            )

    articles = pd.DataFrame(mapped_rows).dropna(subset=["date"]).copy()
    if articles.empty:
        return articles, pd.DataFrame()

    articles["date"] = pd.to_datetime(articles["date"]).dt.date
    articles["is_positive"] = (articles["sentiment_score"] > 0.1).astype(int)
    articles["is_negative"] = (articles["sentiment_score"] < -0.1).astype(int)

    grouped = (
        articles.groupby(["ticker", "date"], as_index=False)
        .agg(
            article_count=("title", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            positive_ratio=("is_positive", "mean"),
            negative_ratio=("is_negative", "mean"),
            avg_relevance=("relevance_score", "mean"),
        )
        .fillna(0.0)
    )
    return articles, grouped


def fetch_prices(tickers: list[str], years: int) -> pd.DataFrame:
    period = f"{years}y"
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    rows: list[dict] = []
    for ticker in tickers:
        try:
            ticker_df = data[ticker].dropna(subset=["Close"]).copy()
        except Exception:
            continue
        if ticker_df.empty:
            continue
        ticker_df = ticker_df.reset_index()
        for _, r in ticker_df.iterrows():
            rows.append(
                {
                    "ticker": ticker,
                    "date": pd.to_datetime(r["Date"]).date(),
                    "open": float(r["Open"]),
                    "high": float(r["High"]),
                    "low": float(r["Low"]),
                    "close": float(r["Close"]),
                    "adj_close": float(r.get("Adj Close", r["Close"])),
                    "volume": float(r["Volume"]),
                }
            )
    return pd.DataFrame(rows)


def add_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    prices = prices.sort_values(["ticker", "date"]).copy()

    def per_ticker(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ticker_value = str(df["ticker"].iloc[0]) if "ticker" in df.columns and not df.empty else ""
        df["daily_return"] = df["close"].pct_change()
        df["returns_5d"] = (df["close"] / df["close"].shift(5) - 1.0) * 100
        df["returns_10d"] = (df["close"] / df["close"].shift(10) - 1.0) * 100
        df["returns_20d"] = (df["close"] / df["close"].shift(20) - 1.0) * 100
        df["volatility_20d"] = df["daily_return"].rolling(20).std()
        df["volume_ratio"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["sma_cross"] = (df["close"].rolling(5).mean() > df["close"].rolling(20).mean()).astype(int)

        df["future_5d_return"] = (df["close"].shift(-5) / df["close"] - 1.0) * 100
        df["future_5d_direction"] = (df["future_5d_return"] > 0).astype(int)
        df["ticker"] = ticker_value
        return df

    features = prices.groupby("ticker", group_keys=False).apply(per_ticker).reset_index(drop=True)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def build_final_dataset(price_features: pd.DataFrame, news_daily: pd.DataFrame) -> pd.DataFrame:
    if price_features.empty:
        return pd.DataFrame()
    if "ticker" not in price_features.columns or "date" not in price_features.columns:
        return pd.DataFrame()
    if news_daily.empty or "ticker" not in news_daily.columns or "date" not in news_daily.columns:
        news_daily = pd.DataFrame(columns=["ticker", "date"])
    merged = price_features.merge(news_daily, how="left", on=["ticker", "date"])
    for col in ["article_count", "avg_sentiment", "sentiment_std", "positive_ratio", "negative_ratio", "avg_relevance"]:
        if col in merged:
            merged[col] = merged[col].fillna(0.0)

    required = [
        "returns_5d",
        "returns_10d",
        "returns_20d",
        "volatility_20d",
        "volume_ratio",
        "rsi_14",
        "sma_cross",
        "future_5d_return",
        "future_5d_direction",
    ]
    cleaned = merged.dropna(subset=required).copy()
    return cleaned


def run(output_root: Path, years: int, rss_lookback_days: int, tickers: list[str]) -> BuildStats:
    stats = BuildStats()
    dirs = ensure_dirs(output_root)

    # 1) Collect RSS/news raw
    news_raw = collect_rss_news(tickers, lookback_days=rss_lookback_days)
    stats.raw_news_rows = int(len(news_raw))
    news_raw.to_csv(dirs["raw"] / "rss_articles_raw.csv", index=False)

    # 2) Build mapped/feature news tables
    articles_mapped, news_daily = build_news_features(news_raw, tickers)
    stats.clean_news_rows = int(len(articles_mapped))
    articles_mapped.to_csv(dirs["processed"] / "news_articles_mapped.csv", index=False)
    news_daily.to_csv(dirs["processed"] / "news_daily_features.csv", index=False)

    # 3) Collect prices
    prices_raw = fetch_prices(tickers, years=years)
    stats.raw_price_rows = int(len(prices_raw))
    prices_raw.to_csv(dirs["raw"] / "yahoo_prices_raw.csv", index=False)

    # 4) Engineer price features
    price_features = add_price_features(prices_raw)
    stats.feature_price_rows = int(len(price_features))
    price_features.to_csv(dirs["processed"] / "price_features.csv", index=False)

    # 5) Merge final dataset
    final_df = build_final_dataset(price_features, news_daily)
    stats.final_rows = int(len(final_df))
    final_df.to_csv(dirs["final"] / "final_model_dataset.csv", index=False)

    # 6) Metadata summary
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker_count": len(tickers),
        "years_of_prices": years,
        "rss_lookback_days": rss_lookback_days,
        "raw_news_rows": stats.raw_news_rows,
        "clean_news_rows": stats.clean_news_rows,
        "raw_price_rows": stats.raw_price_rows,
        "feature_price_rows": stats.feature_price_rows,
        "final_rows_after_cleaning": stats.final_rows,
        "meets_50k_requirement": stats.final_rows >= 50000,
        "final_column_count": int(final_df.shape[1]) if not final_df.empty else 0,
    }
    pd.DataFrame([summary]).to_csv(dirs["final"] / "dataset_summary.csv", index=False)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local stock/news dataset from RSS + Yahoo data.")
    parser.add_argument("--years", type=int, default=8, help="Years of Yahoo price history to pull.")
    parser.add_argument("--rss-lookback-days", type=int, default=3650, help="Google News RSS lookback days in query.")
    parser.add_argument(
        "--ticker-source",
        choices=["seed", "sp500", "nasdaq100", "all"],
        default="sp500",
        help="How to build the ticker universe for local dataset generation.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Optional extra tickers to include, comma-separated.",
    )
    parser.add_argument(
        "--tickers-file",
        type=str,
        default="",
        help="Optional path to a newline-delimited ticker file.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    manual_tickers = _dedupe_keep_order(args.tickers.split(",")) if args.tickers else []
    tickers_file = Path(args.tickers_file).expanduser().resolve() if args.tickers_file else None
    tickers = resolve_ticker_universe(
        ticker_source=args.ticker_source,
        manual_tickers=manual_tickers,
        tickers_file=tickers_file,
    )
    stats = run(
        output_root=root,
        years=args.years,
        rss_lookback_days=args.rss_lookback_days,
        tickers=tickers,
    )
    print(f"Ticker source: {args.ticker_source}")
    print(f"Tickers used: {len(tickers)}")
    print(f"Raw news rows: {stats.raw_news_rows}")
    print(f"Mapped/clean news rows: {stats.clean_news_rows}")
    print(f"Raw price rows: {stats.raw_price_rows}")
    print(f"Price feature rows: {stats.feature_price_rows}")
    print(f"Final merged rows after cleaning: {stats.final_rows}")
    print("Output written under data/raw, data/processed, data/final")


if __name__ == "__main__":
    main()

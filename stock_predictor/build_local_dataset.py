"""Build a large local training dataset from RSS news + Yahoo prices (Polars pipelines)."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import numpy as np
import polars as pl
import requests
from bs4 import BeautifulSoup

from price_fetcher import yfinance_daily_pl


DEFAULT_SEED_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "ABBV", "PFE", "KO",
    "AVGO", "COST", "PEP", "MRK", "BAC", "WMT", "CSCO", "ADBE", "NFLX", "TMO",
    "ACN", "CRM", "MCD", "DHR", "DIS", "ABT", "VZ", "INTC", "CMCSA", "WFC",
    "AMD", "TXN", "TSM", "LIN", "NEE", "BMY", "PM", "UNP", "HON", "QCOM", "LOW",
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
    "TSM": ["taiwan semiconductor", "tsmc", "taiwan semi"],
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

_WIKI_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; stock-predictor-dataset/1.0)"}


@dataclass
class BuildStats:
    raw_news_rows: int = 0
    clean_news_rows: int = 0
    raw_price_rows: int = 0
    feature_price_rows: int = 0
    final_rows: int = 0
    company_tickers_json_path: str = ""
    company_tickers_loaded: int = 0


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


def company_tickers_json_candidates() -> list[Path]:
    """Search order for SEC company ticker JSON (same schema as company_tickers.json)."""
    env = os.environ.get("COMPANY_TICKERS_JSON", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser().resolve())
    candidates.append(Path(__file__).resolve().parent / "data" / "reference" / "company_tickers.json")
    candidates.append(Path("/Users/raghav/company_tickers.json"))
    candidates.append(Path.home() / "company_tickers.json")
    return candidates


def resolve_company_tickers_json_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        return p if p.is_file() else None
    for p in company_tickers_json_candidates():
        if p.is_file():
            return p
    return None


def load_company_tickers_json(path: Path) -> list[str]:
    """Load tickers from SEC-style company_tickers.json: dict of {index: {ticker, title, cik_str}}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    symbols: list[str] = []
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, dict):
                t = v.get("ticker") or v.get("symbol")
                if t:
                    symbols.append(str(t).strip())
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                t = item.get("ticker") or item.get("symbol")
                if t:
                    symbols.append(str(t).strip())
            elif isinstance(item, str):
                symbols.append(item.strip())
    return _dedupe_keep_order(symbols)


def _parse_wikitable_symbols(html: str, table_id: str | None, ticker_header: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=table_id) if table_id else None
    if table is None:
        for cand in soup.find_all("table", class_=lambda c: c and "wikitable" in str(c).lower()):
            headers = [th.get_text(strip=True) for th in cand.find_all("th")]
            if ticker_header in headers:
                table = cand
                break
    if table is None:
        return []
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    try:
        idx = headers.index(ticker_header)
    except ValueError:
        idx = 0
    symbols: list[str] = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) > idx:
            sym = tds[idx].get_text(strip=True).replace(".", "-")
            if sym and re.match(r"^[A-Z0-9.-]{1,15}$", sym, re.I):
                symbols.append(sym.upper())
    return _dedupe_keep_order(symbols)


def fetch_sp500_tickers() -> list[str]:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = requests.get(url, headers=_WIKI_HEADERS, timeout=45)
        r.raise_for_status()
        return _parse_wikitable_symbols(r.text, "constituents", "Symbol")
    except Exception:
        return []


def fetch_nasdaq100_tickers() -> list[str]:
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        r = requests.get(url, headers=_WIKI_HEADERS, timeout=45)
        r.raise_for_status()
        return _parse_wikitable_symbols(r.text, None, "Ticker")
    except Exception:
        return []


def resolve_ticker_universe(
    ticker_source: str,
    manual_tickers: list[str],
    tickers_file: Path | None,
    min_count: int = 50,
    company_json_path: Path | None = None,
) -> tuple[list[str], str, int]:
    """Return (tickers, company_json_path_used, company_json_count)."""
    source_tickers: list[str] = []
    company_path_used = ""
    company_loaded = 0

    json_path = resolve_company_tickers_json_path(company_json_path)
    company_list = load_company_tickers_json(json_path) if json_path else []

    if ticker_source == "company_json":
        if not company_list:
            print(
                "Warning: company_tickers.json not found or empty; "
                f"tried {', '.join(str(p) for p in company_tickers_json_candidates()[:4])}… "
                "falling back to DEFAULT_SEED_TICKERS."
            )
            source_tickers = list(DEFAULT_SEED_TICKERS)
        else:
            source_tickers = company_list
            company_path_used = str(json_path) if json_path else ""
            company_loaded = len(company_list)
    elif ticker_source == "sp500":
        source_tickers = fetch_sp500_tickers()
    elif ticker_source == "nasdaq100":
        source_tickers = fetch_nasdaq100_tickers()
    elif ticker_source == "all":
        source_tickers = (
            company_list
            + fetch_sp500_tickers()
            + fetch_nasdaq100_tickers()
            + list(DEFAULT_SEED_TICKERS)
        )
        if json_path:
            company_path_used = str(json_path)
            company_loaded = len(company_list)
    else:
        source_tickers = list(DEFAULT_SEED_TICKERS)

    file_tickers = load_tickers_from_file(tickers_file) if tickers_file else []
    combined = _dedupe_keep_order(source_tickers + file_tickers + manual_tickers)

    if len(combined) < min_count:
        combined = _dedupe_keep_order(combined + list(DEFAULT_SEED_TICKERS))
    return combined, company_path_used, company_loaded


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


def rss_for_ticker(
    ticker: str,
    lookback_days: int,
    timeout: int = 20,
    max_query_templates: int | None = None,
) -> list[dict]:
    aliases = TICKER_ALIASES.get(ticker, [])
    primary_name = aliases[0] if aliases else ticker
    query_templates = [
        f"{ticker} stock when:{lookback_days}d",
        f"{primary_name} stock when:{lookback_days}d",
        f"{primary_name} earnings guidance when:{lookback_days}d",
        f"{primary_name} lawsuit investigation risk when:{lookback_days}d",
        f"{primary_name} analyst target outlook when:{lookback_days}d",
    ]
    if max_query_templates is not None:
        query_templates = query_templates[: max(1, min(5, int(max_query_templates)))]
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


def collect_rss_news(
    tickers: Iterable[str],
    lookback_days: int,
    max_query_templates: int | None = None,
) -> pl.DataFrame:
    rows: list[dict] = []
    for ticker in tickers:
        rows.extend(
            rss_for_ticker(ticker, lookback_days=lookback_days, max_query_templates=max_query_templates)
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "ticker_seed": pl.Utf8,
                "query_template": pl.Utf8,
                "title": pl.Utf8,
                "url": pl.Utf8,
                "source": pl.Utf8,
                "published_at": pl.Utf8,
                "description": pl.Utf8,
            }
        )
    df = pl.DataFrame(rows)
    df = df.unique(subset=["url", "title"], keep="first")
    pub = pl.col("published_at")
    date_from_iso = pub.str.to_datetime(time_zone="UTC", strict=False).dt.date()
    date_from_prefix = pub.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
    return df.with_columns(
        pl.when(pub.str.len_chars() > 8)
        .then(pl.coalesce(date_from_iso, date_from_prefix))
        .otherwise(None)
        .alias("date")
    ).drop_nulls("date")


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


def build_news_features(news_df: pl.DataFrame, tickers: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    if news_df.is_empty():
        empty_a = pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "date": pl.Date,
                "title": pl.Utf8,
                "url": pl.Utf8,
                "source": pl.Utf8,
                "description": pl.Utf8,
                "highlight": pl.Utf8,
                "sentiment_score": pl.Float64,
                "relevance_score": pl.Float64,
            }
        )
        empty_d = pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "date": pl.Date,
                "article_count": pl.Int64,
                "avg_sentiment": pl.Float64,
                "sentiment_std": pl.Float64,
                "positive_ratio": pl.Float64,
                "negative_ratio": pl.Float64,
                "avg_relevance": pl.Float64,
            }
        )
        return empty_a, empty_d

    mapped_rows: list[dict] = []
    for row in news_df.iter_rows(named=True):
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

    articles = pl.DataFrame(mapped_rows).drop_nulls("date")
    if articles.is_empty():
        return articles, pl.DataFrame()

    articles = articles.with_columns(
        (pl.col("sentiment_score") > 0.1).cast(pl.Int8).alias("is_positive"),
        (pl.col("sentiment_score") < -0.1).cast(pl.Int8).alias("is_negative"),
    )

    grouped = (
        articles.group_by(["ticker", "date"])
        .agg(
            pl.len().alias("article_count"),
            pl.col("sentiment_score").mean().alias("avg_sentiment"),
            pl.col("sentiment_score").std().fill_null(0.0).alias("sentiment_std"),
            pl.col("is_positive").mean().alias("positive_ratio"),
            pl.col("is_negative").mean().alias("negative_ratio"),
            pl.col("relevance_score").mean().alias("avg_relevance"),
        )
        .sort(["ticker", "date"])
    )
    return articles, grouped


def _fetch_one_ticker_history(ticker: str, period: str) -> pl.DataFrame | None:
    try:
        hist = yfinance_daily_pl(ticker, period)
    except ValueError:
        return None
    if hist.is_empty():
        return None
    return hist.with_columns(pl.lit(ticker).alias("ticker"))


def fetch_prices(tickers: list[str], years: int, price_workers: int = 6) -> pl.DataFrame:
    """Download daily OHLCV; uses a small thread pool to overlap Yahoo requests."""
    period = f"{years}y"
    frames: list[pl.DataFrame] = []
    tickers_u = [t for t in tickers if t]
    if not tickers_u:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "date": pl.Date,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    workers = max(1, min(int(price_workers), len(tickers_u), 12))
    if workers == 1:
        for ticker in tickers_u:
            h = _fetch_one_ticker_history(ticker, period)
            if h is not None:
                frames.append(h)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_one_ticker_history, t, period): t for t in tickers_u}
            for fut in as_completed(futures):
                h = fut.result()
                if h is not None:
                    frames.append(h)

    if not frames:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "date": pl.Date,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    return pl.concat(frames, how="vertical")


def add_price_features(prices: pl.DataFrame) -> pl.DataFrame:
    if prices.is_empty():
        return prices
    g = "ticker"
    c = pl.col("close")
    v = pl.col("volume")
    dr = c.pct_change()
    gain = c.diff().clip(lower_bound=0.0).rolling_mean(14, min_samples=1).over(g)
    loss = (-c.diff().clip(upper_bound=0.0)).rolling_mean(14, min_samples=1).over(g)
    rs = gain / pl.max_horizontal(loss, pl.lit(1e-12))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    fut = ((c.shift(-5).over(g) / c) - 1.0) * 100.0

    out = (
        prices.sort([g, "date"])
        .with_columns(
            dr.over(g).alias("daily_return"),
            (((c / c.shift(5).over(g)) - 1.0) * 100.0).alias("returns_5d"),
            (((c / c.shift(10).over(g)) - 1.0) * 100.0).alias("returns_10d"),
            (((c / c.shift(20).over(g)) - 1.0) * 100.0).alias("returns_20d"),
            dr.rolling_std(20, min_samples=2).over(g).alias("volatility_20d"),
            (v.rolling_mean(5, min_samples=1).over(g) / pl.max_horizontal(v.rolling_mean(20, min_samples=1).over(g), pl.lit(1e-9))).alias("volume_ratio"),
            rsi.fill_nan(50.0).alias("rsi_14"),
            (c.rolling_mean(5, min_samples=1).over(g) > c.rolling_mean(20, min_samples=1).over(g))
            .cast(pl.Int8)
            .alias("sma_cross"),
            fut.alias("future_5d_return"),
        )
        .with_columns((pl.col("future_5d_return") > 0).cast(pl.Int8).alias("future_5d_direction"))
    )
    finite_cols = [
        "daily_return",
        "returns_5d",
        "returns_10d",
        "returns_20d",
        "volatility_20d",
        "volume_ratio",
        "rsi_14",
        "future_5d_return",
    ]
    for name in finite_cols:
        out = out.with_columns(
            pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
        )
    return out


def build_final_dataset(price_features: pl.DataFrame, news_daily: pl.DataFrame) -> pl.DataFrame:
    if price_features.is_empty():
        return price_features
    if "ticker" not in price_features.columns or "date" not in price_features.columns:
        return pl.DataFrame()
    if news_daily.is_empty():
        news_daily = pl.DataFrame(schema={"ticker": pl.Utf8, "date": pl.Date})

    merged = price_features.join(news_daily, on=["ticker", "date"], how="left")
    fill_cols = [
        "article_count",
        "avg_sentiment",
        "sentiment_std",
        "positive_ratio",
        "negative_ratio",
        "avg_relevance",
    ]
    for col in fill_cols:
        if col in merged.columns:
            merged = merged.with_columns(pl.col(col).fill_null(0.0))

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
    return merged.drop_nulls(subset=required)


def run(
    output_root: Path,
    years: int,
    rss_lookback_days: int,
    tickers: list[str],
    company_json_meta: tuple[str, int] = ("", 0),
    rss_query_templates: int | None = None,
    price_workers: int = 6,
) -> BuildStats:
    stats = BuildStats()
    stats.company_tickers_json_path, stats.company_tickers_loaded = company_json_meta
    dirs = ensure_dirs(output_root)

    news_raw = collect_rss_news(
        tickers, lookback_days=rss_lookback_days, max_query_templates=rss_query_templates
    )
    stats.raw_news_rows = int(news_raw.height)
    news_raw.write_csv(dirs["raw"] / "rss_articles_raw.csv")

    articles_mapped, news_daily = build_news_features(news_raw, tickers)
    stats.clean_news_rows = int(articles_mapped.height)
    articles_mapped.write_csv(dirs["processed"] / "news_articles_mapped.csv")
    news_daily.write_csv(dirs["processed"] / "news_daily_features.csv")

    prices_raw = fetch_prices(tickers, years=years, price_workers=price_workers)
    stats.raw_price_rows = int(prices_raw.height)
    prices_raw.write_csv(dirs["raw"] / "yahoo_prices_raw.csv")

    price_features = add_price_features(prices_raw)
    stats.feature_price_rows = int(price_features.height)
    price_features.write_csv(dirs["processed"] / "price_features.csv")

    final_df = build_final_dataset(price_features, news_daily)
    stats.final_rows = int(final_df.height)
    final_df.write_csv(dirs["final"] / "final_model_dataset.csv")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker_count": len(tickers),
        "years_of_prices": years,
        "rss_lookback_days": rss_lookback_days,
        "company_tickers_json_path": stats.company_tickers_json_path,
        "company_tickers_json_symbols": stats.company_tickers_loaded,
        "raw_news_rows": stats.raw_news_rows,
        "clean_news_rows": stats.clean_news_rows,
        "raw_price_rows": stats.raw_price_rows,
        "feature_price_rows": stats.feature_price_rows,
        "final_rows_after_cleaning": stats.final_rows,
        "meets_50k_requirement": stats.final_rows >= 50000,
        "final_column_count": int(final_df.width) if not final_df.is_empty() else 0,
    }
    pl.DataFrame([summary]).write_csv(dirs["final"] / "dataset_summary.csv")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local stock/news dataset from RSS + Yahoo data.")
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Years of Yahoo price history (default 5 for faster builds; use 8+ for more history).",
    )
    parser.add_argument(
        "--rss-lookback-days",
        type=int,
        default=365,
        help="Google News 'when:Xd' window per query (default 365; use 3650 for ~10y recall, much slower).",
    )
    parser.add_argument(
        "--rss-query-templates",
        type=int,
        default=2,
        metavar="N",
        help="Use first N RSS query templates per ticker (1-5). Default 2 is fast; 5 matches older full recall.",
    )
    parser.add_argument(
        "--price-workers",
        type=int,
        default=6,
        help="Parallel Yahoo downloads (1-12). Higher can speed builds but may hit rate limits.",
    )
    parser.add_argument(
        "--ticker-source",
        choices=["company_json", "seed", "sp500", "nasdaq100", "all"],
        default="seed",
        help=(
            "Ticker universe for this build only (not the CLI query tickers). "
            "seed = ~90 built-in names (default, fastest). "
            "sp500 ~500 names. company_json ~10k (slow; use --max-tickers). "
            "all = company_json + indices + seed."
        ),
    )
    parser.add_argument(
        "--company-tickers-json",
        type=str,
        default="",
        help="Path to company_tickers.json (overrides env COMPANY_TICKERS_JSON and built-in search paths).",
    )
    parser.add_argument("--tickers", type=str, default="", help="Optional extra tickers, comma-separated.")
    parser.add_argument("--tickers-file", type=str, default="", help="Optional newline-delimited ticker file.")
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="If >0, cap universe to the first N symbols after merge (for smoke tests). 0 = no cap.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    manual_tickers = _dedupe_keep_order(args.tickers.split(",")) if args.tickers else []
    tickers_file = Path(args.tickers_file).expanduser().resolve() if args.tickers_file else None
    company_arg = Path(args.company_tickers_json).expanduser().resolve() if args.company_tickers_json else None
    tickers, company_path_used, company_loaded = resolve_ticker_universe(
        ticker_source=args.ticker_source,
        manual_tickers=manual_tickers,
        tickers_file=tickers_file,
        company_json_path=company_arg,
    )
    if args.max_tickers and args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]
    rss_q = max(1, min(5, int(args.rss_query_templates)))
    stats = run(
        output_root=root,
        years=args.years,
        rss_lookback_days=args.rss_lookback_days,
        tickers=tickers,
        company_json_meta=(company_path_used, company_loaded),
        rss_query_templates=rss_q,
        price_workers=max(1, min(12, int(args.price_workers))),
    )
    print(f"Ticker source: {args.ticker_source}")
    if company_path_used:
        print(f"Company tickers JSON: {company_path_used} ({company_loaded} symbols before merge)")
    print(f"Tickers used: {len(tickers)}")
    print(f"Raw news rows: {stats.raw_news_rows}")
    print(f"Mapped/clean news rows: {stats.clean_news_rows}")
    print(f"Raw price rows: {stats.raw_price_rows}")
    print(f"Price feature rows: {stats.feature_price_rows}")
    print(f"Final merged rows after cleaning: {stats.final_rows}")
    print("Output written under data/raw, data/processed, data/final")


if __name__ == "__main__":
    main()

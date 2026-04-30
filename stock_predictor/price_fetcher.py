"""Yahoo Finance integration and technical feature engineering (Polars)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import yfinance as yf


def _rsi_expr(close: pl.Expr, window: int = 14) -> pl.Expr:
    delta = close.diff()
    gain = delta.clip(lower_bound=0.0).rolling_mean(window_size=window, min_samples=1)
    loss = (-delta.clip(upper_bound=0.0)).rolling_mean(window_size=window, min_samples=1)
    rs = gain / loss.replace(0.0, None)
    return (100.0 - (100.0 / (1.0 + rs))).fill_nan(50.0)


def _finite_float(value: object, default: float) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
        if not np.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def compute_price_features(history_df: pl.DataFrame) -> dict[str, float]:
    """Engineer latest-row technical features from sorted OHLCV history (Polars)."""
    if history_df.is_empty():
        return {
            "returns_5d": 0.0,
            "returns_10d": 0.0,
            "returns_20d": 0.0,
            "volatility_20d": 0.0,
            "volume_ratio": 1.0,
            "rsi_14": 50.0,
            "sma_cross": 0.0,
            "current_price": 0.0,
        }

    df = history_df.sort("date")
    close = df["close"]
    volume = df["volume"]
    n = df.height

    def _pct_ret(lookback: int) -> float:
        if n <= lookback:
            return 0.0
        last = float(close[-1])
        prev = float(close[-(lookback + 1)])
        if prev == 0.0:
            return 0.0
        return float((last / prev - 1.0) * 100.0)

    daily_ret = close.pct_change()
    vol_20 = _finite_float(
        daily_ret.tail(20).std() if n > 20 else daily_ret.std(),
        0.0,
    )

    v5 = float(volume.tail(5).mean()) if n >= 5 else float(volume.mean() or 0.0)
    v20 = float(volume.tail(20).mean()) if n >= 20 else float(volume.mean() or 0.0)
    volume_ratio = (v5 / v20) if v20 else 1.0

    rsi_df = df.select(_rsi_expr(pl.col("close"), 14).alias("rsi"))
    rsi_14 = _finite_float(rsi_df["rsi"][-1], 50.0) if n > 14 else 50.0

    sma5 = float(close.tail(5).mean()) if n >= 5 else float(close[-1])
    sma20 = float(close.tail(20).mean()) if n >= 20 else float(close[-1])
    sma_cross = 1.0 if sma5 > sma20 else 0.0

    return {
        "returns_5d": _pct_ret(5),
        "returns_10d": _pct_ret(10),
        "returns_20d": _pct_ret(20),
        "volatility_20d": vol_20,
        "volume_ratio": float(volume_ratio),
        "rsi_14": rsi_14,
        "sma_cross": sma_cross,
        "current_price": float(close[-1]),
    }


def yfinance_daily_pl(ticker: str, period: str) -> pl.DataFrame:
    """Expose daily OHLCV as Polars for dataset builds (same path as live fetch)."""
    return _yf_history_to_polars(ticker, period)


def _yf_history_to_polars(ticker: str, period: str) -> pl.DataFrame:
    """Build Polars frame from yfinance without pandas→polars conversion (no pyarrow)."""
    hist = yf.Ticker(ticker).history(period=period, interval="1d")
    if hist.empty:
        raise ValueError(f"No price data found for {ticker}")

    def _d(x) -> list:
        return [xi.date() if hasattr(xi, "date") else xi for xi in x]

    dates = _d(hist.index)
    return pl.DataFrame(
        {
            "date": dates,
            "open": hist["Open"].to_numpy(dtype=np.float64, copy=False),
            "high": hist["High"].to_numpy(dtype=np.float64, copy=False),
            "low": hist["Low"].to_numpy(dtype=np.float64, copy=False),
            "close": hist["Close"].to_numpy(dtype=np.float64, copy=False),
            "volume": hist["Volume"].to_numpy(dtype=np.float64, copy=False),
        }
    ).drop_nulls(subset=["close", "volume"])


def fetch_price_data(ticker: str, period_days: int = 60) -> dict[str, Any]:
    """Return current price, historical Polars frame, and engineered features."""
    period = f"{max(30, period_days + 40)}d"
    history = _yf_history_to_polars(ticker, period)
    features = compute_price_features(history)
    return {"ticker": ticker, "history_df": history, **features}

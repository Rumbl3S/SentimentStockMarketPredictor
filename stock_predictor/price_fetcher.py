"""Yahoo Finance integration and technical feature engineering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_price_features(history_df: pd.DataFrame) -> dict[str, float]:
    close = history_df["Close"]
    volume = history_df["Volume"]

    daily_returns = close.pct_change()
    returns_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0.0
    returns_10d = (
        float((close.iloc[-1] / close.iloc[-11] - 1) * 100) if len(close) > 10 else 0.0
    )
    returns_20d = (
        float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 20 else 0.0
    )
    volatility_20d = float(daily_returns.rolling(20).std().iloc[-1]) if len(close) > 20 else 0.0
    vol_5 = float(volume.tail(5).mean()) if len(volume) >= 5 else float(volume.mean())
    vol_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
    volume_ratio = (vol_5 / vol_20) if vol_20 else 1.0
    rsi_14 = float(_compute_rsi(close, 14).iloc[-1]) if len(close) > 14 else 50.0
    sma_5 = float(close.tail(5).mean()) if len(close) >= 5 else float(close.iloc[-1])
    sma_20 = float(close.tail(20).mean()) if len(close) >= 20 else float(close.iloc[-1])
    sma_cross = 1.0 if sma_5 > sma_20 else 0.0

    return {
        "returns_5d": returns_5d,
        "returns_10d": returns_10d,
        "returns_20d": returns_20d,
        "volatility_20d": volatility_20d,
        "volume_ratio": float(volume_ratio),
        "rsi_14": rsi_14,
        "sma_cross": sma_cross,
        "current_price": float(close.iloc[-1]),
    }


def fetch_price_data(ticker: str, period_days: int = 60) -> dict[str, Any]:
    """Return current price, historical data, and engineered features."""
    period = f"{max(30, period_days + 40)}d"
    history = yf.Ticker(ticker).history(period=period, interval="1d")
    if history.empty:
        raise ValueError(f"No price data found for {ticker}")

    history = history.dropna(subset=["Close", "Volume"])
    features = compute_price_features(history)
    return {"ticker": ticker, "history_df": history, **features}

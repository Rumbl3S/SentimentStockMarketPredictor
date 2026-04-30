"""Train-only feature preprocessing: winsorization and correlation pruning."""

from __future__ import annotations

import numpy as np


def winsorize_fit(X: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-column percentile bounds from training matrix only."""
    low = np.percentile(X, lower_pct, axis=0)
    high = np.percentile(X, upper_pct, axis=0)
    return low.astype(np.float64), high.astype(np.float64)


def winsorize_apply(X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(X.astype(np.float64), low, high)


def correlation_pruning_mask(X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Return boolean mask of columns to keep; drops later column of each highly correlated pair (train only)."""
    if X.shape[1] <= 1:
        return np.ones(X.shape[1], dtype=bool)
    # Correlation across samples; suppress warnings for constant cols
    with np.errstate(invalid="ignore"):
        corr = np.abs(np.corrcoef(X.astype(np.float64), rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    p = X.shape[1]
    keep = np.ones(p, dtype=bool)
    for i in range(p):
        if not keep[i]:
            continue
        for j in range(i + 1, p):
            if not keep[j]:
                continue
            if corr[i, j] >= threshold:
                keep[j] = False
    return keep


def apply_column_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return X[:, mask]


def class_imbalance_report(y: np.ndarray) -> dict[str, float | bool]:
    """Summarize binary label balance for training diagnostics."""
    y = y.astype(int)
    n = len(y)
    if n == 0:
        return {
            "n_samples": 0.0,
            "positive_count": 0.0,
            "negative_count": 0.0,
            "minority_class_fraction": 0.0,
            "imbalance_severe": False,
        }
    pos = int((y == 1).sum())
    neg = n - pos
    minority = min(pos, neg)
    frac = minority / max(1, n)
    return {
        "n_samples": float(n),
        "positive_count": float(pos),
        "negative_count": float(neg),
        "minority_class_fraction": float(frac),
        "imbalance_severe": bool(frac < 0.35),
    }

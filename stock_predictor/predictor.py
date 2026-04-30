"""ML prediction: Polars price frames, train-only preprocessing, HP tuning, imbalance-aware metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_preprocessing import (
    apply_column_mask,
    class_imbalance_report,
    correlation_pruning_mask,
    winsorize_apply,
    winsorize_fit,
)
from price_fetcher import compute_price_features


class StockPredictor:
    def __init__(self) -> None:
        self._rf_best: Pipeline | None = None
        self._ridge_best: Pipeline | None = None
        self._winsor_low: np.ndarray | None = None
        self._winsor_high: np.ndarray | None = None
        self._feature_mask: np.ndarray | None = None

    @staticmethod
    def _sentiment_vector(sentiment_features: dict[str, Any]) -> list[float]:
        return [
            float(sentiment_features.get("finbert_avg", 0.0)),
            float(sentiment_features.get("marketaux_avg", 0.0)),
            float(sentiment_features.get("composite_score", 0.0)),
            float(sentiment_features.get("positive_ratio", 0.0)),
            float(sentiment_features.get("negative_ratio", 0.0)),
            float(sentiment_features.get("max_sentiment", 0.0)),
            float(sentiment_features.get("min_sentiment", 0.0)),
            float(sentiment_features.get("sentiment_std", 0.0)),
        ]

    @staticmethod
    def _sanitize_vector(values: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def build_training_data(
        self, price_df: pl.DataFrame, sentiment_features: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows: list[list[float]] = []
        y_class: list[int] = []
        y_reg: list[float] = []
        sentiment_vec = self._sentiment_vector(sentiment_features)
        df = price_df.sort("date")
        n = df.height

        for idx in range(20, n - 5):
            sub_df = df.slice(0, idx + 1)
            price_feats = compute_price_features(sub_df)
            feature_row = sentiment_vec + [
                float(price_feats["returns_5d"]),
                float(price_feats["returns_10d"]),
                float(price_feats["returns_20d"]),
                float(price_feats["volatility_20d"]),
                float(price_feats["volume_ratio"]),
                float(price_feats["rsi_14"]),
                float(price_feats["sma_cross"]),
            ]
            current_close = float(sub_df["close"][-1])
            future_close = float(df.slice(idx + 5, 1)["close"][0])
            future_pct_change = ((future_close / current_close) - 1.0) * 100.0

            safe_row = self._sanitize_vector(feature_row).tolist()
            rows.append(safe_row)
            y_class.append(1 if future_pct_change > 0 else 0)
            y_reg.append(future_pct_change)

        if not rows:
            return np.array([]), np.array([]), np.array([])
        X = np.array(rows, dtype=float)
        y_c = np.array(y_class, dtype=float)
        y_r = np.array(y_reg, dtype=float)
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y_c) & np.isfinite(y_r)
        return X[valid_mask], y_c[valid_mask].astype(int), y_r[valid_mask]

    def _apply_fitted_preprocess(self, X: np.ndarray) -> np.ndarray:
        if self._winsor_low is None or self._winsor_high is None or self._feature_mask is None:
            return X
        Xw = winsorize_apply(X, self._winsor_low, self._winsor_high)
        return apply_column_mask(Xw, self._feature_mask)

    @staticmethod
    def _close_vol_stats(price_df: pl.DataFrame) -> tuple[float, float]:
        sorted_df = price_df.sort("date")
        ch5 = sorted_df.select(pl.col("close").pct_change(5)).drop_nulls()
        five_std = float(ch5.to_series().std()) if ch5.height else 0.0
        daily = sorted_df.select(pl.col("close").pct_change()).drop_nulls()["close"]
        recent_std = float(daily.tail(20).std()) if daily.len() > 0 else 0.0
        if recent_std <= 0 or np.isnan(recent_std):
            recent_std = float(daily.std()) if daily.len() > 0 else 0.0
        return five_std, recent_std

    @staticmethod
    def blend_prediction(
        model_direction: str,
        model_magnitude: float,
        model_confidence: float,
        rf_score: float,
        lr_r2: float,
        composite_sentiment: float,
        price_df: pl.DataFrame,
        score_is_f1: bool,
    ) -> dict[str, Any]:
        """Blend model output with sentiment; rf_score is accuracy or F1 depending on imbalance."""
        model_reliable = rf_score >= 0.55 and lr_r2 >= -1.0
        baseline = 0.5 if not score_is_f1 else 0.45
        model_weight = max(min((rf_score - baseline) * 2, 0.8), 0.2)
        sentiment_weight = 1.0 - model_weight

        five_std, recent_daily_std = StockPredictor._close_vol_stats(price_df)
        clamp_bound = max(0.0, 2 * five_std * 100)
        if not model_reliable and clamp_bound > 0:
            model_magnitude = float(np.clip(model_magnitude, -clamp_bound, +clamp_bound))

        five_day_expected_move = max(0.0, recent_daily_std * np.sqrt(5) * 100)
        sentiment_magnitude = composite_sentiment * five_day_expected_move * 2.0
        blended_magnitude = (model_weight * model_magnitude) + (sentiment_weight * sentiment_magnitude)
        final_direction = "UP" if blended_magnitude >= 0 else "DOWN"

        if (model_direction == "UP" and composite_sentiment >= 0) or (
            model_direction == "DOWN" and composite_sentiment < 0
        ):
            signal_msg = "Sentiment and price momentum AGREE"
            mixed_signals = False
        else:
            signal_msg = "Low Confidence - Mixed Signals"
            mixed_signals = True

        return {
            "direction": final_direction,
            "magnitude_pct": abs(float(blended_magnitude)),
            "confidence": float(model_confidence),
            "signal": signal_msg,
            "mixed_signals": mixed_signals,
            "model_weight_used": model_weight,
            "sentiment_weight_used": sentiment_weight,
            "model_reliable": model_reliable,
            "clamp_bound_pct": float(clamp_bound),
            "model_raw_direction": model_direction,
            "model_raw_magnitude_pct": float(model_magnitude),
            "sentiment_magnitude_pct": float(sentiment_magnitude),
        }

    def predict(
        self,
        features: np.ndarray,
        sentiment_data: dict[str, Any],
        rf_score: float,
        lr_r2: float,
        price_df: pl.DataFrame,
        score_is_f1: bool,
    ) -> dict[str, Any]:
        if self._rf_best is None or self._ridge_best is None:
            raise RuntimeError("Models are not fitted.")

        x_proc = self._apply_fitted_preprocess(self._sanitize_vector(features).reshape(1, -1))
        class_pred = int(self._rf_best.predict(x_proc)[0])
        probs = self._rf_best.predict_proba(x_proc)[0]
        confidence = float(max(probs))
        model_magnitude = float(self._ridge_best.predict(x_proc)[0])
        direction = "UP" if class_pred == 1 else "DOWN"
        composite = float(sentiment_data.get("composite_score", 0.0))
        return self.blend_prediction(
            model_direction=direction,
            model_magnitude=model_magnitude,
            model_confidence=confidence,
            rf_score=rf_score,
            lr_r2=lr_r2,
            composite_sentiment=composite,
            price_df=price_df,
            score_is_f1=score_is_f1,
        )

    def _tune_models(
        self,
        X_train: np.ndarray,
        y_class_train: np.ndarray,
        y_reg_train: np.ndarray,
        imbalance_severe: bool,
    ) -> None:
        n = len(X_train)
        cv = min(5, max(2, n // 15))
        cv = max(2, cv)

        rf_base = RandomForestClassifier(random_state=42)
        rf_pipe = Pipeline([("scaler", StandardScaler()), ("clf", rf_base)])
        rf_grid = {
            "clf__n_estimators": [80, 120, 200],
            "clf__max_depth": [4, 8, 12, 16, None],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__min_samples_split": [2, 4],
            "clf__class_weight": [None, "balanced", "balanced_subsample"],
        }
        rf_scoring = "f1" if imbalance_severe else "f1_macro"
        if n >= 40:
            rf_search = RandomizedSearchCV(
                rf_pipe,
                rf_grid,
                n_iter=14,
                cv=cv,
                scoring=rf_scoring,
                random_state=42,
                n_jobs=1,
            )
            rf_search.fit(X_train, y_class_train)
            self._rf_best = rf_search.best_estimator_
        else:
            self._rf_best = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=120,
                            max_depth=8,
                            min_samples_leaf=2,
                            class_weight="balanced" if imbalance_severe else None,
                            random_state=42,
                        ),
                    ),
                ]
            )
            self._rf_best.fit(X_train, y_class_train)

        ridge_pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
        ridge_grid = {"reg__alpha": np.logspace(-2, 2, num=12)}
        if n >= 40:
            ridge_search = RandomizedSearchCV(
                ridge_pipe,
                ridge_grid,
                n_iter=10,
                cv=cv,
                scoring="neg_mean_absolute_error",
                random_state=42,
                n_jobs=1,
            )
            ridge_search.fit(X_train, y_reg_train)
            self._ridge_best = ridge_search.best_estimator_
        else:
            self._ridge_best = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=1.0)),
                ]
            )
            self._ridge_best.fit(X_train, y_reg_train)

    def run_prediction(
        self, price_data: dict[str, Any], sentiment_results: dict[str, Any]
    ) -> dict[str, Any]:
        price_df = price_data["history_df"]
        if not isinstance(price_df, pl.DataFrame):
            price_df = pl.from_pandas(price_df)

        X, y_class, y_reg = self.build_training_data(price_df, sentiment_results)
        if len(X) < 10:
            composite = float(sentiment_results.get("composite_score", 0.0))
            _, recent_daily_std = self._close_vol_stats(price_df)
            five_day_expected_move = max(0.0, recent_daily_std * np.sqrt(5) * 100)
            fallback_direction = "UP" if composite >= 0 else "DOWN"
            fallback_magnitude = abs(composite * five_day_expected_move * 2.0)
            return {
                "direction": fallback_direction,
                "magnitude_pct": fallback_magnitude,
                "confidence": 0.5,
                "signal": "Insufficient price history - sentiment-led estimate",
                "mixed_signals": False,
                "model_weight_used": 0.0,
                "sentiment_weight_used": 1.0,
                "model_reliable": False,
                "clamp_bound_pct": 0.0,
                "rf_accuracy": None,
                "rf_f1": None,
                "lr_r2": None,
                "lr_mae": None,
                "class_imbalance": None,
                "training_samples": int(len(X)),
                "preprocessing": {
                    "winsorize": False,
                    "correlation_pruning": False,
                    "dropped_feature_count": 0,
                    "imbalance_severe": False,
                },
            }

        split_idx = max(1, int(len(X) * 0.7))
        split_idx = min(split_idx, len(X) - 1)

        X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
        y_class_train, y_class_test = y_class[:split_idx], y_class[split_idx:]
        y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]

        imb = class_imbalance_report(y_class_train)
        imbalance_severe = bool(imb["imbalance_severe"])

        w_low, w_high = winsorize_fit(X_train_raw)
        X_train_w = winsorize_apply(X_train_raw, w_low, w_high)
        X_test_w = winsorize_apply(X_test_raw, w_low, w_high)

        mask = correlation_pruning_mask(X_train_w, threshold=0.95)
        dropped = int((~mask).sum())
        X_train = apply_column_mask(X_train_w, mask)
        X_test = apply_column_mask(X_test_w, mask)

        self._winsor_low = w_low
        self._winsor_high = w_high
        self._feature_mask = mask.astype(bool)

        self._tune_models(X_train, y_class_train, y_reg_train, imbalance_severe)

        class_test_pred = self._rf_best.predict(X_test)
        reg_test_pred = self._ridge_best.predict(X_test)

        rf_accuracy = float(accuracy_score(y_class_test, class_test_pred))
        rf_f1 = float(
            f1_score(y_class_test, class_test_pred, average="binary", pos_label=1, zero_division=0)
        )
        lr_r2 = float(r2_score(y_reg_test, reg_test_pred))
        lr_mae = float(mean_absolute_error(y_reg_test, reg_test_pred))

        rf_score_for_blend = rf_f1 if imbalance_severe else rf_accuracy

        latest_features = self._sanitize_vector(
            self._sentiment_vector(sentiment_results)
            + [
                float(price_data["returns_5d"]),
                float(price_data["returns_10d"]),
                float(price_data["returns_20d"]),
                float(price_data["volatility_20d"]),
                float(price_data["volume_ratio"]),
                float(price_data["rsi_14"]),
                float(price_data["sma_cross"]),
            ]
        )
        prediction = self.predict(
            latest_features,
            sentiment_results,
            rf_score=rf_score_for_blend,
            lr_r2=lr_r2,
            price_df=price_df,
            score_is_f1=imbalance_severe,
        )
        prediction["rf_accuracy"] = rf_accuracy
        prediction["rf_f1"] = rf_f1
        prediction["lr_r2"] = lr_r2
        prediction["lr_mae"] = lr_mae
        prediction["class_imbalance"] = imb
        prediction["training_samples"] = int(len(X))
        prediction["preprocessing"] = {
            "winsorize": True,
            "correlation_pruning": True,
            "dropped_feature_count": dropped,
            "imbalance_severe": imbalance_severe,
        }
        return prediction

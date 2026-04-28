"""ML prediction module combining sentiment and technical features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

from price_fetcher import compute_price_features


class StockPredictor:
    def __init__(self) -> None:
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lr_regressor = Ridge(alpha=1.0)
        self.scaler = StandardScaler()

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

    def build_training_data(
        self, price_df: pd.DataFrame, sentiment_features: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build feature matrix and class/regression labels from historical prices."""
        rows: list[list[float]] = []
        y_class: list[int] = []
        y_reg: list[float] = []
        sentiment_vec = self._sentiment_vector(sentiment_features)

        for idx in range(20, len(price_df) - 5):
            sub_df = price_df.iloc[: idx + 1]
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
            current_close = float(price_df["Close"].iloc[idx])
            future_close = float(price_df["Close"].iloc[idx + 5])
            future_pct_change = ((future_close / current_close) - 1) * 100

            rows.append(feature_row)
            y_class.append(1 if future_pct_change > 0 else 0)
            y_reg.append(future_pct_change)

        if not rows:
            return np.array([]), np.array([]), np.array([])
        return np.array(rows), np.array(y_class), np.array(y_reg)

    def train(
        self,
        X_train: np.ndarray,
        y_class_train: np.ndarray,
        y_reg_train: np.ndarray,
    ) -> None:
        """Train classifier and regressor on scaled features."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.rf_classifier.fit(X_train_scaled, y_class_train)
        self.lr_regressor.fit(X_train_scaled, y_reg_train)

    @staticmethod
    def blend_prediction(
        model_direction: str,
        model_magnitude: float,
        model_confidence: float,
        rf_accuracy: float,
        lr_r2: float,
        composite_sentiment: float,
        price_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Blend model output with sentiment, weighted by model reliability."""
        model_reliable = rf_accuracy >= 0.55 and lr_r2 >= -1.0

        # Continuous confidence-based weighting:
        # 0.50 RF acc -> 0.20 model weight, 0.90+ -> 0.80 model weight.
        model_weight = max(min((rf_accuracy - 0.5) * 2, 0.8), 0.2)
        sentiment_weight = 1.0 - model_weight

        five_day_returns = price_df["Close"].pct_change(5).dropna()
        return_std = float(five_day_returns.std()) if not five_day_returns.empty else 0.0
        clamp_bound = max(0.0, 2 * return_std * 100)
        if not model_reliable and clamp_bound > 0:
            model_magnitude = float(np.clip(model_magnitude, -clamp_bound, clamp_bound))

        daily_returns = price_df["Close"].pct_change().dropna()
        recent_daily_std = (
            float(daily_returns.tail(20).std())
            if not daily_returns.empty
            else 0.0
        )
        if np.isnan(recent_daily_std) or recent_daily_std <= 0:
            recent_daily_std = float(daily_returns.std()) if not daily_returns.empty else 0.0
        five_day_expected_move = max(0.0, recent_daily_std * np.sqrt(5) * 100)
        sentiment_magnitude = composite_sentiment * five_day_expected_move * 2.0
        blended_magnitude = (model_weight * model_magnitude) + (
            sentiment_weight * sentiment_magnitude
        )
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
        rf_accuracy: float,
        lr_r2: float,
        price_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate direction + magnitude prediction from one feature vector."""
        x_scaled = self.scaler.transform(features.reshape(1, -1))
        class_pred = int(self.rf_classifier.predict(x_scaled)[0])
        probs = self.rf_classifier.predict_proba(x_scaled)[0]
        confidence = float(max(probs))
        model_magnitude = float(self.lr_regressor.predict(x_scaled)[0])
        direction = "UP" if class_pred == 1 else "DOWN"

        composite = float(sentiment_data.get("composite_score", 0.0))
        return self.blend_prediction(
            model_direction=direction,
            model_magnitude=model_magnitude,
            model_confidence=confidence,
            rf_accuracy=rf_accuracy,
            lr_r2=lr_r2,
            composite_sentiment=composite,
            price_df=price_df,
        )

    def run_prediction(
        self, price_data: dict[str, Any], sentiment_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Build data, train models, evaluate metrics, and predict latest move."""
        price_df = price_data["history_df"]
        X, y_class, y_reg = self.build_training_data(price_df, sentiment_results)
        if len(X) < 10:
            composite = float(sentiment_results.get("composite_score", 0.0))
            daily_returns = price_df["Close"].pct_change().dropna()
            recent_daily_std = (
                float(daily_returns.tail(20).std()) if not daily_returns.empty else 0.0
            )
            if np.isnan(recent_daily_std) or recent_daily_std <= 0:
                recent_daily_std = float(daily_returns.std()) if not daily_returns.empty else 0.0
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
                "lr_r2": None,
                "training_samples": int(len(X)),
            }

        split_idx = max(1, int(len(X) * 0.7))
        split_idx = min(split_idx, len(X) - 1)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_class_train, y_class_test = y_class[:split_idx], y_class[split_idx:]
        y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]

        self.train(X_train, y_class_train, y_reg_train)
        x_test_scaled = self.scaler.transform(X_test)
        class_test_pred = self.rf_classifier.predict(x_test_scaled)
        reg_test_pred = self.lr_regressor.predict(x_test_scaled)

        rf_accuracy = float(accuracy_score(y_class_test, class_test_pred))
        lr_r2 = float(r2_score(y_reg_test, reg_test_pred))

        latest_features = np.array(
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
            rf_accuracy=rf_accuracy,
            lr_r2=lr_r2,
            price_df=price_df,
        )
        prediction["rf_accuracy"] = rf_accuracy
        prediction["lr_r2"] = lr_r2
        prediction["training_samples"] = int(len(X))
        return prediction

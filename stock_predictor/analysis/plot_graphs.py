"""Build analysis plots from generated datasets (Polars read + matplotlib).

Plots 1–5 are the original course-style figures. Plot 6 compares direction models
(RandomForest vs dummy vs logistic) when ``model_metrics.csv`` is present.
Plot 7 is a KMeans elbow curve on article embedding columns (same inputs as PCA clustering).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def _embedding_matrix(df_articles: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    embed_cols = [c for c in df_articles.columns if c.startswith("embedding_")]
    if not embed_cols:
        return np.array([]), []
    x = df_articles.select(embed_cols).to_numpy()
    return x, embed_cols


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (11, 6)


def plot_finbert_distribution(df_articles: pl.DataFrame, out_dir: Path) -> None:
    order = ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"]
    plt.figure()
    colors = sns.color_palette("RdYlGn", n_colors=len(order))
    for i, label in enumerate(order):
        vals = df_articles.filter(pl.col("sentiment_label") == label)["overall_sentiment_score"].to_numpy()
        if len(vals):
            plt.hist(vals, bins=40, range=(-1, 1), alpha=0.45, label=label, color=colors[i], stacked=False)
    plt.title("FinBERT Sentiment Score Distribution by Label")
    plt.xlabel("overall_sentiment_score")
    plt.ylabel("Article count")
    plt.xlim(-1, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "1_finbert_sentiment_distribution.png", dpi=180)
    plt.close()


def plot_similarity_distribution(df_articles: pl.DataFrame, out_dir: Path, similarity_floor: float) -> None:
    plt.figure()
    vals = df_articles["similarity_score"].to_numpy()
    plt.hist(vals, bins=45, color="#4c78a8")
    plt.axvline(
        x=similarity_floor,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"similarity_floor={similarity_floor:.2f}",
    )
    plt.title("Cosine Similarity Distribution with Similarity Floor")
    plt.xlabel("similarity_score")
    plt.ylabel("Article count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "2_similarity_floor_distribution.png", dpi=180)
    plt.close()


def plot_embedding_clusters(df_articles: pl.DataFrame, out_dir: Path, n_clusters: int) -> None:
    embed_cols = [c for c in df_articles.columns if c.startswith("embedding_")]
    x = df_articles.select(embed_cols).to_numpy()
    pca = PCA(n_components=2, random_state=42)
    x2 = pca.fit_transform(x)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(x)

    n = min(2400, len(cluster_labels))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(cluster_labels), size=n, replace=False) if len(cluster_labels) > n else np.arange(len(cluster_labels))
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        x2[idx, 0],
        x2[idx, 1],
        c=cluster_labels[idx],
        cmap="tab10",
        alpha=0.7,
        s=28,
    )
    plt.colorbar(scatter, label="cluster_id")
    plt.title("Article Embedding Clusters (PCA to 2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(out_dir / "3_embedding_clusters_pca.png", dpi=180)
    plt.close()


def plot_rf_importances(df_importances: pl.DataFrame, out_dir: Path) -> None:
    plot_df = df_importances.sort("importance")
    plt.figure(figsize=(10, 6))
    feats = plot_df["feature"].to_list()
    imps = plot_df["importance"].to_numpy()
    plt.barh(feats, imps, color=sns.color_palette("Blues_r", n_colors=len(feats)))
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "4_rf_feature_importances.png", dpi=180)
    plt.close()


def plot_semantic_vs_sentiment(df_event_ticker: pl.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(10, 7))
    up = df_event_ticker.filter(pl.col("predicted_direction") == "UP")
    down = df_event_ticker.filter(pl.col("predicted_direction") == "DOWN")
    if up.height:
        plt.scatter(
            up["semantic_score"].to_numpy(),
            up["sentiment_score"].to_numpy(),
            c="#2ca02c",
            marker="o",
            s=100,
            alpha=0.8,
            label="UP",
        )
    if down.height:
        plt.scatter(
            down["semantic_score"].to_numpy(),
            down["sentiment_score"].to_numpy(),
            c="#d62728",
            marker="s",
            s=100,
            alpha=0.8,
            label="DOWN",
        )
    plt.axhline(0.0, color="gray", linewidth=1, alpha=0.6)
    plt.axvline(0.0, color="gray", linewidth=1, alpha=0.6)
    plt.title("Semantic Score vs. Sentiment Score (Per Ticker)")
    plt.xlabel("semantic_score")
    plt.ylabel("sentiment_score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "5_semantic_vs_sentiment_scatter.png", dpi=180)
    plt.close()


def plot_model_comparison(df_metrics: pl.DataFrame, out_dir: Path) -> bool:
    """Bar chart: mean accuracy / F1 / ROC-AUC by model (teammate CSV schema)."""
    metric_cols = [c for c in ("accuracy", "f1", "roc_auc") if c in df_metrics.columns]
    if not metric_cols or "model" not in df_metrics.columns:
        return False
    summary = (
        df_metrics.group_by("model")
        .agg([pl.mean(c).alias(c) for c in metric_cols])
        .sort("model")
    )
    long_df = summary.unpivot(
        index=["model"],
        on=metric_cols,
        variable_name="metric",
        value_name="score",
    ).filter(pl.col("score").is_finite())
    if long_df.is_empty():
        return False

    plt.figure(figsize=(11, 7))
    sns.barplot(data=pd.DataFrame(long_df.to_dicts()), x="metric", y="score", hue="model")
    plt.ylim(0.0, 1.0)
    plt.title("Model evaluation comparison (mean across event/ticker splits)")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "6_model_comparison_metrics.png", dpi=180)
    plt.close()
    return True


def plot_elbow_method(df_articles: pl.DataFrame, out_dir: Path, max_k: int = 10) -> bool:
    """Elbow curve for KMeans inertia on article embedding matrix."""
    x, _ = _embedding_matrix(df_articles)
    if x.size == 0 or len(x) < 15:
        return False
    k_hi = min(max(2, int(max_k)), len(x) - 1, 15)
    ks: list[int] = []
    inertias: list[float] = []
    for k in range(1, k_hi + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(x)
        ks.append(k)
        inertias.append(float(km.inertia_))
    plt.figure(figsize=(9, 5.5))
    plt.plot(ks, inertias, marker="o", color="#4c78a8")
    plt.xlabel("k (clusters)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("KMeans elbow on article embeddings")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "7_elbow_method.png", dpi=180)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis graphs.")
    parser.add_argument("--data-dir", default="analysis/data", help="Directory with CSV inputs.")
    parser.add_argument("--output-dir", default="analysis/plots", help="Directory for plot images.")
    parser.add_argument("--similarity-floor", type=float, default=0.35)
    parser.add_argument("--n-clusters", type=int, default=4, help="KMeans clusters for embedding PCA plot.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_articles = pl.read_csv(data_dir / "articles.csv")
    df_event_ticker = pl.read_csv(data_dir / "event_ticker_scores.csv")
    df_importances = pl.read_csv(data_dir / "rf_feature_importances.csv")
    model_metrics_path = data_dir / "model_metrics.csv"
    df_model_metrics = (
        pl.read_csv(model_metrics_path) if model_metrics_path.exists() else pl.DataFrame()
    )

    _set_style()
    plot_finbert_distribution(df_articles, out_dir)
    plot_similarity_distribution(df_articles, out_dir, args.similarity_floor)
    plot_embedding_clusters(df_articles, out_dir, args.n_clusters)
    plot_rf_importances(df_importances, out_dir)
    plot_semantic_vs_sentiment(df_event_ticker, out_dir)
    n_plots = 5
    if plot_elbow_method(df_articles, out_dir, max_k=10):
        n_plots += 1
    if not df_model_metrics.is_empty() and "model" in df_model_metrics.columns:
        if plot_model_comparison(df_model_metrics, out_dir):
            n_plots += 1
    print(f"Saved {n_plots} plot file(s) under {out_dir} (plots 1–5; plus 6–7 when data allows).")


if __name__ == "__main__":
    main()

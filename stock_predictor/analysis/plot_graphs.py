"""Build five analysis plots from generated datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (11, 6)


def plot_finbert_distribution(df_articles: pd.DataFrame, out_dir: Path) -> None:
    order = ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"]
    plt.figure()
    sns.histplot(
        data=df_articles,
        x="overall_sentiment_score",
        hue="sentiment_label",
        hue_order=order,
        bins=40,
        multiple="stack",
        palette="RdYlGn",
    )
    plt.title("FinBERT Sentiment Score Distribution by Label")
    plt.xlabel("overall_sentiment_score")
    plt.ylabel("Article count")
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "1_finbert_sentiment_distribution.png", dpi=180)
    plt.close()


def plot_similarity_distribution(
    df_articles: pd.DataFrame, out_dir: Path, similarity_floor: float
) -> None:
    plt.figure()
    sns.histplot(df_articles["similarity_score"], bins=45, color="#4c78a8")
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


def plot_embedding_clusters(df_articles: pd.DataFrame, out_dir: Path, n_clusters: int) -> None:
    embed_cols = [c for c in df_articles.columns if c.startswith("embedding_")]
    x = df_articles[embed_cols]
    pca = PCA(n_components=2, random_state=42)
    x2 = pca.fit_transform(x)

    # Recluster from embeddings so chart demonstrates end-to-end unsupervised grouping.
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(x)

    plot_df = pd.DataFrame({"pc1": x2[:, 0], "pc2": x2[:, 1], "cluster_id": cluster_labels})
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df.sample(min(2400, len(plot_df)), random_state=42),
        x="pc1",
        y="pc2",
        hue="cluster_id",
        palette="tab10",
        alpha=0.7,
        s=28,
    )
    plt.title("Article Embedding Clusters (PCA to 2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="cluster_id", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "3_embedding_clusters_pca.png", dpi=180)
    plt.close()


def plot_rf_importances(df_importances: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df_importances.sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        x="importance",
        y="feature",
        hue="feature",
        palette="Blues_r",
        dodge=False,
        legend=False,
    )
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "4_rf_feature_importances.png", dpi=180)
    plt.close()


def plot_semantic_vs_sentiment(df_event_ticker: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_event_ticker,
        x="semantic_score",
        y="sentiment_score",
        hue="predicted_direction",
        style="predicted_direction",
        palette={"UP": "#2ca02c", "DOWN": "#d62728"},
        s=100,
        alpha=0.8,
    )
    plt.axhline(0.0, color="gray", linewidth=1, alpha=0.6)
    plt.axvline(0.0, color="gray", linewidth=1, alpha=0.6)
    plt.title("Semantic Score vs. Sentiment Score (Per Ticker)")
    plt.xlabel("semantic_score")
    plt.ylabel("sentiment_score")
    plt.tight_layout()
    plt.savefig(out_dir / "5_semantic_vs_sentiment_scatter.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis graphs.")
    parser.add_argument("--data-dir", default="analysis/data", help="Directory with CSV inputs.")
    parser.add_argument("--output-dir", default="analysis/plots", help="Directory for plot images.")
    parser.add_argument("--similarity-floor", type=float, default=0.35)
    parser.add_argument("--n-clusters", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_articles = pd.read_csv(data_dir / "articles.csv")
    df_event_ticker = pd.read_csv(data_dir / "event_ticker_scores.csv")
    df_importances = pd.read_csv(data_dir / "rf_feature_importances.csv")

    _set_style()
    plot_finbert_distribution(df_articles, out_dir)
    plot_similarity_distribution(df_articles, out_dir, args.similarity_floor)
    plot_embedding_clusters(df_articles, out_dir, args.n_clusters)
    plot_rf_importances(df_importances, out_dir)
    plot_semantic_vs_sentiment(df_event_ticker, out_dir)
    print(f"Saved 5 plots in {out_dir}")


if __name__ == "__main__":
    main()

# ML-Powered Stock Sentiment and Prediction Pipeline

This project includes:
- A Python CLI pipeline for stock sentiment and price-direction prediction.
- A lightweight Flask API (`/api/predict`) exposing the same pipeline.
- A React frontend with two interfaces:
  - input page for model parameters
  - results page with ranking, sentiment, and blend breakdown

## Setup

1. Create and activate a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Build the local dataset first (Polars pipelines + RSS + Yahoo).
   - **Important:** `main.py` / the API only load news for **the tickers the user types**. The slow step is **`build_local_dataset.py`**, which pre-fetches RSS + prices for the **universe** you choose.
   - **Defaults are tuned for a quick build** (~tens of minutes on a typical connection): `seed` (~90 tickers), **5 years** of prices, **365-day** RSS window, **2** RSS query templates per ticker, **6** parallel Yahoo workers. Example:
     - `./.venv/bin/python build_local_dataset.py`
   - **Heavier / older-style full S&P + long lookback** (much longer):
     - `./.venv/bin/python build_local_dataset.py --ticker-source sp500 --years 8 --rss-lookback-days 3650 --rss-query-templates 5`
   - Full SEC list: `--ticker-source company_json` (optionally `--max-tickers 200` for a smoke build).
4. (Optional) copy `.env.example` to `.env` for local overrides.

## Data stack: Polars (not pandas)

Tabular work in this repo uses **Polars** for I/O, joins, group-bys, and feature engineering. Yahoo Finance still returns a pandas object internally; we immediately copy **Open/High/Low/Close/Volume** into Polars so the project’s own logic stays in Polars end-to-end. `yfinance` may still install pandas as a transitive dependency, but **application code** uses Polars for tables.

## Usage

### CLI

Interactive mode:

`python main.py`

Direct CLI args:

`python main.py --query "How will AI chip demand affect semiconductor stocks?" --tickers "NVDA,TSM,INTC" --pages 3 --history-days 180`

Optional clustering:

`python main.py --query "How will tariffs affect chipmakers?" --tickers "NVDA,AMD,INTC" --cluster`

### API

Run backend API:

`python api_server.py`

Endpoints:
- `GET /api/health`
- `POST /api/predict`

Example request body:

```json
{
  "query": "How will AI chip demand affect semiconductor stocks?",
  "tickers": "NVDA,TSM,INTC",
  "pages": 3,
  "historyDays": 180,
  "cluster": true
}
```

### Frontend

1. Install frontend dependencies:
   - `cd frontend`
   - `npm install`
2. Start frontend:
   - `npm run dev`
3. Optional API base override:
   - create `frontend/.env` with `VITE_API_BASE_URL=http://localhost:8000`

## Analysis graph scripts

Generate synthetic test data (includes `model_metrics.csv` for model comparison):

`python analysis/generate_test_data.py --output-dir analysis/data --seed 42`

Create analysis graphs (plots 1–5 always; **6** and **7** when inputs allow):

`python analysis/plot_graphs.py --data-dir analysis/data --output-dir analysis/plots --similarity-floor 0.35` (embedding KMeans defaults to **5** clusters; override with `--n-clusters`.)

Outputs:
- `analysis/plots/1_finbert_sentiment_distribution.png`
- `analysis/plots/2_similarity_floor_distribution.png`
- `analysis/plots/3_embedding_clusters_pca.png`
- `analysis/plots/4_rf_feature_importances.png`
- `analysis/plots/5_semantic_vs_sentiment_scatter.png`
- `analysis/plots/6_model_comparison_metrics.png` — mean accuracy / F1 / ROC-AUC for **RandomForest (current)**, **Dummy uniform baseline**, and **LogisticRegression**, when `analysis/data/model_metrics.csv` exists (synthetic generator writes it; real runs use `generate_real_data.py`).
- `analysis/plots/7_elbow_method.png` — KMeans inertia vs. `k` on article embedding columns (requires enough articles with embeddings).

Optional **live** CSVs (RSS/API + Yahoo; slower), same column layout as synthetic plus `model_metrics.csv`:

`python analysis/generate_real_data.py --output-dir analysis/data_real`

Then:

`python analysis/plot_graphs.py --data-dir analysis/data_real --output-dir analysis/plots_real --similarity-floor 0.35`

## Pipeline architecture (7 steps)

1. Read user query and ticker list.
2. Extract keywords from the query.
3. Fetch ticker-specific news from the **local** processed dataset (`news_articles_mapped.csv`).
4. Rank articles using TF-IDF + cosine similarity.
5. Run FinBERT sentiment and combine with **dataset** keyword/entity-style scores (still labeled `marketaux_avg` in JSON for backward compatibility).
6. Pull historical price data and engineer technical features (**Polars** in `price_fetcher.py`).
7. **Preprocess**, **tune hyperparameters** (train only), evaluate with **multiple metrics**, then blend predictions for the summary.

## Pre-processing, feature engineering, tuning, and metrics

| Area | Before | After |
|------|--------|--------|
| **Tables** | pandas-heavy pipelines | **Polars** for build, news read, prices, joins, and aggregations. |
| **Outliers / heavy tails** | ad hoc `nan_to_num` only | **Winsorization** at the 1st/99th percentile **fit on the training split only**, then applied to train, test, and the live feature row—reduces leverage from extreme returns/volume without peeking at the test split. |
| **Redundant features** | all columns always used | **Correlation pruning** on the training matrix (drop one column of each pair with \|r\| ≥ 0.95) to reduce multicollinearity before scaling/tuning. |
| **Scaling** | `StandardScaler` fit on train | unchanged principle: scaling lives **inside** each sklearn `Pipeline` during `RandomizedSearchCV`, so CV folds do not leak statistics from validation folds. |
| **Hyperparameters** | fixed RF / Ridge | **`RandomizedSearchCV`** on the **training** portion only (time-ordered 70%): RF grid over depth, leaves, `class_weight`, estimators; Ridge over `logspace` alphas. Small training sets fall back to sensible defaults to avoid unstable CV. |
| **Class imbalance** | ignored | **Report** minority fraction on the training labels; if the minority share is **below 35%**, treat as “severe”: RF search optimizes **F1**, `class_weight` options include **balanced**, and the sentiment blend uses **F1** instead of raw accuracy as the reliability score. |
| **Metrics** | accuracy + R² only | Still report those, plus **binary F1** for direction (better when UP/DOWN counts differ) and **MAE** for the Ridge target (5-day % move in percentage points—interpretable error size). |

Together: preprocessing fit only on training data, imbalance-aware tuning when labels are skewed, multiple metrics (accuracy, F1, MAE, R² as applicable), and hyperparameter search aligned with those objectives.

## ML concepts used

- Polars for ETL-style transforms
- TF-IDF + cosine similarity ranking
- FinBERT (`ProsusAI/finbert`)
- Random Forest (direction) + Ridge (magnitude), with **RandomizedSearchCV**
- Train-only winsorization + correlation pruning + `Pipeline` + `StandardScaler`
- Optional K-Means article clustering (`--cluster`)

## Limitations and disclaimer

- News quality depends on RSS coverage and ticker/alias matching.
- Per-ticker **live** training rows depend on `--history-days` and data availability; the README no longer claims a fixed “~40 rows” cap.
- Historical rows still reuse **current-query** sentiment features for practicality (documented limitation for true walk-forward forecasting).
- This is a class project and not financial advice.

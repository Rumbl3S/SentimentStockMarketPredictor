# ML-Powered Stock Sentiment and Prediction Pipeline

This project includes:
- A Python CLI pipeline for stock sentiment and price-direction prediction.
- A lightweight Flask API (`/api/predict`) exposing the same pipeline.
- A React frontend with two interfaces:
  - input page for model parameters
  - transparency-focused results page showing ranking, sentiment, and blend math

## Setup

1. Create and activate a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Build the local dataset first:
   - `./.venv/bin/python build_local_dataset.py --ticker-source all --years 8 --rss-lookback-days 3650`
4. (Optional) copy `.env.example` to `.env` for local overrides.

## Usage

### CLI

Interactive mode:

`python main.py`

Direct CLI args:

`python main.py --query "How will AI chip demand affect semiconductor stocks?" --tickers "NVDA,TSM,INTC" --pages 3 --history-days 60`

Optional clustering bonus:

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

## Analysis Graph Scripts

Generate synthetic test data (enough volume for meaningful distributions/clusters):

`python analysis/generate_test_data.py --output-dir analysis/data --seed 42`

Create all five analysis graphs:

`python analysis/plot_graphs.py --data-dir analysis/data --output-dir analysis/plots --similarity-floor 0.35`

Outputs:
- `analysis/plots/1_finbert_sentiment_distribution.png`
- `analysis/plots/2_similarity_floor_distribution.png`
- `analysis/plots/3_embedding_clusters_pca.png`
- `analysis/plots/4_rf_feature_importances.png`
- `analysis/plots/5_semantic_vs_sentiment_scatter.png`

## Pipeline Architecture (7 Steps)

1. Read user query and ticker list.
2. Extract keywords from the query.
3. Fetch ticker-specific news from the local processed dataset.
4. Rank articles using TF-IDF + cosine similarity.
5. Run FinBERT sentiment analysis and combine with MarketAux entity sentiment.
6. Pull historical price data and engineer technical features.
7. Train per-ticker models and print prediction summaries.

## ML Concepts Used

- TF-IDF vectorization
- Cosine similarity ranking
- FinBERT transformer-based sentiment (`ProsusAI/finbert`)
- Random Forest classification (UP/DOWN)
- Linear Regression (% move magnitude)
- Feature engineering across NLP + price signals
- Train/test split and evaluation metrics
- Optional K-Means article clustering (`--cluster`)

## Limitations and Disclaimer

- News quality depends on RSS availability and ticker mention extraction.
- Training windows are small (~40 usable rows per ticker), so predictive power is limited.
- Sentiment in historical training rows is approximated by current sentiment context for class-project practicality.
- This is a class project and not financial advice.

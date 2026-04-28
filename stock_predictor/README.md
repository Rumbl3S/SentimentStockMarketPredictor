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
3. Configure API key:
   - `cp .env.example .env`
   - Put your key in either `MARKETAUX_API_KEY` or `NEWS_API` in `.env`

## Get a MarketAux API Key

Create a free account at [MarketAux](https://www.marketaux.com/) and copy your API token into `.env`.

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

## Pipeline Architecture (7 Steps)

1. Read user query and ticker list.
2. Extract keywords from the query.
3. Fetch ticker-specific paginated news from MarketAux.
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

- MarketAux free tier limits article coverage (3 articles/request, 100 requests/day).
- Training windows are small (~40 usable rows per ticker), so predictive power is limited.
- Sentiment in historical training rows is approximated by current sentiment context for class-project practicality.
- This is a class project and not financial advice.

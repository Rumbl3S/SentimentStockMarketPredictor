# ML-Powered Stock Sentiment and Prediction Pipeline

This project is a Python CLI application that takes a natural-language market question plus stock tickers, fetches ticker-specific news from MarketAux, ranks articles by query relevance with TF-IDF + cosine similarity, analyzes sentiment with FinBERT, engineers price features from Yahoo Finance, and predicts next-5-day direction and magnitude using Random Forest + Linear Regression.

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

Interactive mode:

`python main.py`

Direct CLI args:

`python main.py --query "How will AI chip demand affect semiconductor stocks?" --tickers "NVDA,TSM,INTC" --top-k 5 --pages 3 --history-days 60`

Optional clustering bonus:

`python main.py --query "How will tariffs affect chipmakers?" --tickers "NVDA,AMD,INTC" --cluster`

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

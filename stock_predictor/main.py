"""CLI entrypoint for ML-powered stock sentiment and prediction pipeline."""

from __future__ import annotations

import argparse

from config import HISTORICAL_DAYS
from output_formatter import print_results
from pipeline_runner import run_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock sentiment + prediction pipeline using local RSS-backed news + FinBERT + sklearn."
    )
    parser.add_argument("--query", type=str, help="Natural language query")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers, e.g. NVDA,TSM,INTC")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--history-days", type=int, default=HISTORICAL_DAYS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    query = args.query or input("Enter your query: ").strip()
    tickers_input = args.tickers or input("Enter tickers (comma-separated): ").strip()
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if not query or not tickers:
        print("Query and at least one ticker are required.")
        return

    payload = run_pipeline(
        query=query,
        tickers=tickers,
        pages=args.pages,
        history_days=args.history_days,
        cluster=args.cluster,
    )
    print(f"\nExtracted keywords: {payload['inputs']['keywords']}\n")
    print(f"API search keywords: {payload['inputs']['api_keywords']}\n")

    print_results(
        all_results=payload["per_ticker"],
        query=query,
        api_requests_used=payload["meta"]["api_requests_used"],
        cluster_info=payload["cluster_info"],
    )


if __name__ == "__main__":
    main()

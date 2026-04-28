"""Flask API wrapper for the stock prediction pipeline."""

from __future__ import annotations

from http import HTTPStatus

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import HISTORICAL_DAYS
from pipeline_runner import run_pipeline

app = Flask(__name__)
CORS(app)


@app.get("/api/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, HTTPStatus.OK


@app.post("/api/predict")
def predict() -> tuple[dict, int]:
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    tickers_raw = payload.get("tickers", "")
    if isinstance(tickers_raw, str):
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    elif isinstance(tickers_raw, list):
        tickers = [str(t).strip().upper() for t in tickers_raw if str(t).strip()]
    else:
        tickers = []

    pages = int(payload.get("pages", 3))
    history_days = int(payload.get("historyDays", HISTORICAL_DAYS))
    cluster = bool(payload.get("cluster", False))

    if not query or not tickers:
        return {
            "error": "ValidationError",
            "message": "Query and at least one ticker are required.",
        }, HTTPStatus.BAD_REQUEST

    try:
        result = run_pipeline(
            query=query,
            tickers=tickers,
            pages=pages,
            history_days=history_days,
            cluster=cluster,
        )
    except Exception as exc:
        return {
            "error": "PipelineError",
            "message": str(exc),
        }, HTTPStatus.INTERNAL_SERVER_ERROR

    return jsonify(result), HTTPStatus.OK


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

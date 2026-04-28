import type { SentimentSummary as SentimentSummaryType } from '../types'

export function SentimentSummary({ sentiment }: { sentiment: SentimentSummaryType }) {
  return (
    <section className="card">
      <h3>Sentiment Aggregate</h3>
      <div className="metrics-grid">
        <p className="metric-item"><span>FinBERT avg</span>{sentiment.finbert_avg.toFixed(4)}</p>
        <p className="metric-item"><span>MarketAux avg</span>{sentiment.marketaux_avg.toFixed(4)}</p>
        <p className="metric-item"><span>Composite</span>{sentiment.composite_score.toFixed(4)}</p>
        <p className="metric-item"><span>Positive</span>{sentiment.positive_count}</p>
        <p className="metric-item"><span>Negative</span>{sentiment.negative_count}</p>
        <p className="metric-item"><span>Neutral</span>{sentiment.neutral_count}</p>
        <p className="metric-item"><span>Positive ratio</span>{(sentiment.positive_ratio * 100).toFixed(1)}%</p>
        <p className="metric-item"><span>Negative ratio</span>{(sentiment.negative_ratio * 100).toFixed(1)}%</p>
      </div>
    </section>
  )
}

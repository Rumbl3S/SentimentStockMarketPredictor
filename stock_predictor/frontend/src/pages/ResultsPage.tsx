import { Link, useLocation } from 'react-router-dom'
import { ArticleContributionChart } from '../components/ArticleContributionChart'
import { ArticleRankingTable } from '../components/ArticleRankingTable'
import { ClusterSummary } from '../components/ClusterSummary'
import { ModelBlendBreakdown } from '../components/ModelBlendBreakdown'
import { PredictionCard } from '../components/PredictionCard'
import { PriceFeatureCard } from '../components/PriceFeatureCard'
import { SentimentSummary } from '../components/SentimentSummary'
import { loadLastResult } from './HomePage'
import type { PredictResponse } from '../types'

export function ResultsPage() {
  const location = useLocation()
  const stateData = location.state as PredictResponse | undefined
  const data = stateData ?? loadLastResult()

  if (!data) {
    return (
      <main className="container">
        <section className="card">
          <h1>No Results Yet</h1>
          <p className="subtitle">Run a prediction from the home page first.</p>
          <Link to="/">Go to homepage</Link>
        </section>
      </main>
    )
  }

  return (
    <main className="container">
      <header className="card hero-card">
        <h1>Prediction results</h1>
        <p className="subtitle">Query: {data.inputs.query}</p>
        <p><strong>Keywords:</strong> {data.inputs.keywords.join(', ') || 'none'}</p>
        <p><strong>API search terms:</strong> {data.inputs.api_keywords.join(', ') || 'none'}</p>
        <p><strong>API requests used:</strong> {data.meta.api_requests_used}</p>
        {!!data.meta.warnings.length && <p className="error">Warnings: {data.meta.warnings.join(' | ')}</p>}
        <Link className="inline-link" to="/">Run another prediction</Link>
      </header>

      {Object.entries(data.per_ticker).map(([ticker, result]) => (
        <section key={ticker} className="ticker-section">
          <h2 className="ticker-title">{ticker}</h2>
          <PredictionCard prediction={result.prediction} />
          <ModelBlendBreakdown prediction={result.prediction} />
          <ArticleRankingTable articles={result.articles} />
          <ArticleContributionChart articles={result.sentiment.articles} />
          <SentimentSummary sentiment={result.sentiment} />
          <PriceFeatureCard price={result.price} />
        </section>
      ))}

      <ClusterSummary clusterInfo={data.cluster_info} />
    </main>
  )
}

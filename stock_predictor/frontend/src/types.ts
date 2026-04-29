export type PredictRequest = {
  query: string
  tickers: string
  pages: number
  historyDays: number
  cluster: boolean
}

export type Finbert = {
  score: number
  positive: number
  negative: number
  neutral: number
  label: string
}

export type Article = {
  uuid: string
  ticker: string
  title: string
  description: string
  snippet: string
  url: string
  source: string
  published_at: string
  entity_sentiment_score: number | null
  entity_match_score: number | null
  relevance_score: number
  highlights: { highlight: string; sentiment: string | null; highlighted_in: string }[]
  finbert: Finbert
}

export type SentimentSummary = {
  articles: Article[]
  finbert_avg: number
  marketaux_avg: number
  composite_score: number
  positive_count: number
  negative_count: number
  neutral_count: number
  positive_ratio: number
  negative_ratio: number
  sentiment_std: number
}

export type PriceFeatures = {
  current_price: number
  returns_5d: number
  returns_10d: number
  returns_20d: number
  volatility_20d: number
  volume_ratio: number
  rsi_14: number
  sma_cross: number
}

export type Prediction = {
  direction: 'UP' | 'DOWN'
  magnitude_pct: number
  confidence: number
  signal: string
  mixed_signals: boolean
  model_weight_used: number
  sentiment_weight_used: number
  model_reliable: boolean
  clamp_bound_pct: number
  model_raw_direction?: string
  model_raw_magnitude_pct?: number
  sentiment_magnitude_pct?: number
  rf_accuracy: number | null
  lr_r2: number | null
  training_samples: number
}

export type TickerResult = {
  articles: Article[]
  sentiment: SentimentSummary
  price: PriceFeatures
  prediction: Prediction
}

export type PredictResponse = {
  inputs: {
    query: string
    tickers: string[]
    pages: number
    history_days: number
    top_k: number
    cluster: boolean
    keywords: string[]
    api_keywords: string[]
  }
  per_ticker: Record<string, TickerResult>
  cluster_info: {
    enabled: boolean
    query_cluster: number
    clusters: Record<string, { size: number; top_terms: string[]; top_tickers: [string, number][] }>
  } | null
  meta: {
    api_requests_used: number
    warnings: string[]
    generated_at: string
  }
}

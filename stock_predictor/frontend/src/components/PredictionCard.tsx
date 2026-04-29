import type { Prediction } from '../types'

export function PredictionCard({ prediction }: { prediction: Prediction }) {
  const directionClass = prediction.direction === 'UP' ? 'pill-positive' : 'pill-negative'
  return (
    <section className="card">
      <h3>Prediction Headline</h3>
      <div className="headline-row">
        <p className="headline">
          {prediction.direction === 'UP' ? '+' : '-'}
          {prediction.magnitude_pct.toFixed(2)}%
        </p>
        <span className={`direction-pill ${directionClass}`}>{prediction.direction}</span>
      </div>
      <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(1)}%</p>
      <p><strong>Signal:</strong> {prediction.signal}</p>
    </section>
  )
}

import type { Prediction } from '../types'

function toPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'n/a'
  }
  return `${value.toFixed(2)}%`
}

export function ModelBlendBreakdown({ prediction }: { prediction: Prediction }) {
  return (
    <section className="card">
      <h3>Model Math</h3>
      <div className="metrics-grid">
        <p className="metric-item"><span>Raw model direction</span>{prediction.model_raw_direction ?? 'n/a'}</p>
        <p className="metric-item"><span>Raw model magnitude</span>{toPct(prediction.model_raw_magnitude_pct)}</p>
        <p className="metric-item"><span>Sentiment magnitude</span>{toPct(prediction.sentiment_magnitude_pct)}</p>
        <p className="metric-item"><span>Final blended magnitude</span>{toPct(prediction.magnitude_pct)}</p>
        <p className="metric-item"><span>Model weight</span>{(prediction.model_weight_used * 100).toFixed(1)}%</p>
        <p className="metric-item"><span>Sentiment weight</span>{(prediction.sentiment_weight_used * 100).toFixed(1)}%</p>
        <p className="metric-item"><span>RF accuracy</span>{prediction.rf_accuracy === null ? 'n/a' : prediction.rf_accuracy.toFixed(3)}</p>
        <p className="metric-item"><span>LR R²</span>{prediction.lr_r2 === null ? 'n/a' : prediction.lr_r2.toFixed(3)}</p>
        <p className="metric-item"><span>Training samples</span>{prediction.training_samples}</p>
        <p className="metric-item"><span>Clamp bound</span>{toPct(prediction.clamp_bound_pct)}</p>
      </div>
    </section>
  )
}

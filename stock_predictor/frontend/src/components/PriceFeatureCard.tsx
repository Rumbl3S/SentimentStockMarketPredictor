import type { PriceFeatures } from '../types'

export function PriceFeatureCard({ price }: { price: PriceFeatures }) {
  return (
    <section className="card">
      <h3>Price Features</h3>
      <div className="metrics-grid">
        <p className="metric-item"><span>Current Price</span>${price.current_price.toFixed(2)}</p>
        <p className="metric-item"><span>Returns 5d</span>{price.returns_5d.toFixed(2)}%</p>
        <p className="metric-item"><span>Returns 10d</span>{price.returns_10d.toFixed(2)}%</p>
        <p className="metric-item"><span>Returns 20d</span>{price.returns_20d.toFixed(2)}%</p>
        <p className="metric-item"><span>Volatility 20d</span>{price.volatility_20d.toFixed(4)}</p>
        <p className="metric-item"><span>Volume ratio</span>{price.volume_ratio.toFixed(4)}</p>
        <p className="metric-item"><span>RSI 14</span>{price.rsi_14.toFixed(2)}</p>
        <p className="metric-item"><span>SMA cross</span>{price.sma_cross.toFixed(0)}</p>
      </div>
    </section>
  )
}

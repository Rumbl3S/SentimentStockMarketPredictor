import { useState } from 'react'
import type { PredictRequest } from '../types'

type Props = {
  loading: boolean
  onSubmit: (request: PredictRequest) => Promise<void>
}

const STORAGE_KEY = 'stock_predictor_last_inputs'

function getInitialForm(): PredictRequest {
  const raw = localStorage.getItem(STORAGE_KEY)
  if (!raw) {
    return { query: '', tickers: '', pages: 3, historyDays: 180, cluster: false }
  }
  try {
    return JSON.parse(raw) as PredictRequest
  } catch {
    return { query: '', tickers: '', pages: 3, historyDays: 180, cluster: false }
  }
}

export function InputForm({ loading, onSubmit }: Props) {
  const [form, setForm] = useState<PredictRequest>(() => getInitialForm())
  const [error, setError] = useState('')

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (!form.query.trim() || !form.tickers.trim()) {
      setError('Query and tickers are required.')
      return
    }
    setError('')
    localStorage.setItem(STORAGE_KEY, JSON.stringify(form))
    await onSubmit(form)
  }

  return (
    <form className="card form" onSubmit={submit}>
      <h2>Run Prediction</h2>
      <label>
        Query
        <textarea
          value={form.query}
          onChange={(e) => setForm((prev) => ({ ...prev, query: e.target.value }))}
          placeholder="How will AI chip demand affect semiconductor stocks?"
        />
      </label>
      <label>
        Tickers (comma-separated)
        <input
          value={form.tickers}
          onChange={(e) => setForm((prev) => ({ ...prev, tickers: e.target.value }))}
          placeholder="NVDA,TSM,INTC"
        />
      </label>
      <div className="row">
        <label>
          Pages
          <input
            type="number"
            min={1}
            max={10}
            value={form.pages}
            onChange={(e) => setForm((prev) => ({ ...prev, pages: Number(e.target.value) }))}
          />
        </label>
        <label>
          History Days
          <input
            type="number"
            min={30}
            max={3650}
            value={form.historyDays}
            onChange={(e) => setForm((prev) => ({ ...prev, historyDays: Number(e.target.value) }))}
          />
        </label>
      </div>
      <label className="checkbox">
        <input
          type="checkbox"
          checked={form.cluster}
          onChange={(e) => setForm((prev) => ({ ...prev, cluster: e.target.checked }))}
        />
        Include optional article clustering
      </label>
      {error && <p className="error">{error}</p>}
      <button className="primary-button" type="submit" disabled={loading}>
        {loading ? 'Running model...' : 'Run model'}
      </button>
    </form>
  )
}

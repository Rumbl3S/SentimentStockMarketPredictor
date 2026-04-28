import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { predict } from '../api'
import { InputForm } from '../components/InputForm'
import type { PredictResponse } from '../types'

const RESULT_KEY = 'stock_predictor_last_result'

export function HomePage() {
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  async function handleSubmit(request: {
    query: string
    tickers: string
    pages: number
    historyDays: number
    cluster: boolean
  }) {
    try {
      setLoading(true)
      setError('')
      const data = await predict(request)
      localStorage.setItem(RESULT_KEY, JSON.stringify(data))
      navigate('/results', { state: data })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="container">
      <header className="card hero-card">
        <h1>Stock Sentiment Predictor</h1>
        <p className="subtitle">
          Enter model inputs and run a transparent prediction workflow.
        </p>
      </header>
      <InputForm loading={loading} onSubmit={handleSubmit} />
      {error && <p className="error card error-card">{error}</p>}
    </main>
  )
}

export function loadLastResult(): PredictResponse | null {
  const raw = localStorage.getItem(RESULT_KEY)
  if (!raw) {
    return null
  }
  try {
    return JSON.parse(raw) as PredictResponse
  } catch {
    return null
  }
}

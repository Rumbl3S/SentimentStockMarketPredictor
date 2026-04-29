import type { PredictRequest, PredictResponse } from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export async function predict(payload: PredictRequest): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data?.message ?? 'Prediction request failed')
  }
  return data as PredictResponse
}

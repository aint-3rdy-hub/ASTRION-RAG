export type AskState = 'answer' | 'no-results' | 'error'

export interface Latency {
  retrieval_seconds: number
  generation_seconds: number
  total_seconds: number
}

export interface SourceHit {
  source: string
  page: number
  chunk_id: string
  score: number
  rank: number
}

export interface RetrievedChunk {
  rank: number
  score: number
  source: string
  page: number
  chunk_id: string
  text: string
  vector_id?: number
}

export interface AskResponse {
  state: AskState
  question: string
  answer: string
  sources: SourceHit[]
  retrieved_chunks: RetrievedChunk[]
  retrieval_count: number
  latency: Latency
  error?: string
}

export interface DocumentRow {
  id: string
  filename: string
  pages: number
  chunks: number
  status: string
}

export interface StatsResponse {
  ready: boolean
  documents: number
  chunks: number
  pages: number
  embedding_model: string
  llm_model: string
  vector_store: string
  documents_list: DocumentRow[]
  documents_dir?: string
  groq_ready?: boolean
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export function getStats() {
  return request<StatsResponse>('/api/stats')
}

export function getDocuments() {
  return request<{
    documents: DocumentRow[]
    total_documents: number
    total_pages: number
    total_chunks: number
    ready: boolean
    documents_dir?: string
  }>('/api/documents')
}

export async function uploadDocuments(files: File[]) {
  const body = new FormData()
  for (const file of files) {
    body.append('files', file)
  }
  const response = await fetch('/api/documents/upload', {
    method: 'POST',
    body,
  })
  if (!response.ok) {
    let detail = `Upload failed (${response.status})`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (typeof payload.detail === 'string' && payload.detail.trim()) {
        detail = payload.detail
      }
    } catch {
      // keep status fallback
    }
    throw new Error(detail)
  }
  return response.json() as Promise<{
    saved: string[]
    message: string
    documents: DocumentRow[]
    total_documents: number
    total_pages: number
    total_chunks: number
    ready: boolean
    documents_dir?: string
  }>
}

export function askQuestion(question: string) {
  return request<AskResponse>('/api/ask', {
    method: 'POST',
    body: JSON.stringify({ question }),
  })
}

export function getEvaluation() {
  return request<{
    available: boolean
    message?: string
    metrics: { label: string; value: string; unit: string }[]
    rows: {
      id?: string
      question: string
      expected_source: string
      retrieved_source: string
      result: string
      latency: string | null
      status?: string
    }[]
    failures: {
      id?: string
      title: string
      description: string
      cause: string
      mitigation: string
    }[]
    notes?: string
    groundedness: number | null
    citation_score: number | null
  }>('/api/evaluation')
}

import { useEffect, useRef, useState } from 'react'
import {
  askQuestion,
  type AskResponse,
  type AskState,
  type StatsResponse,
} from '../lib/api'

type UiState = 'empty' | 'loading' | AskState

interface LoadingStage {
  label: string
  done: boolean
}

const EXAMPLE_QUESTIONS = [
  'When is a firewall useful according to the documents?',
  'Why does chunking use overlapping windows?',
  'What should log collection include?',
]

const LOADING_STAGES: LoadingStage[] = [
  { label: 'Retrieving evidence…', done: false },
  { label: 'Building context…', done: false },
  { label: 'Generating answer…', done: false },
]

export default function AskPage({ stats }: { stats: StatsResponse | null }) {
  const [question, setQuestion] = useState('')
  const [submittedQuestion, setSubmittedQuestion] = useState('')
  const [appState, setAppState] = useState<UiState>('empty')
  const [loadingStages, setLoadingStages] = useState<LoadingStage[]>(LOADING_STAGES)
  const [result, setResult] = useState<AskResponse | null>(null)
  const [traceOpen, setTraceOpen] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = async (raw?: string) => {
    const query = (raw ?? question).trim()
    if (!query || appState === 'loading') return

    setSubmittedQuestion(query)
    setQuestion('')
    setAppState('loading')
    setTraceOpen(false)
    setResult(null)
    setLoadingStages(LOADING_STAGES.map((stage) => ({ ...stage, done: false })))

    const t1 = window.setTimeout(() => {
      setLoadingStages((prev) => prev.map((stage, i) => (i === 0 ? { ...stage, done: true } : stage)))
    }, 350)
    const t2 = window.setTimeout(() => {
      setLoadingStages((prev) => prev.map((stage, i) => (i <= 1 ? { ...stage, done: true } : stage)))
    }, 900)

    try {
      const payload = await askQuestion(query)
      window.clearTimeout(t1)
      window.clearTimeout(t2)
      setLoadingStages((prev) => prev.map((stage) => ({ ...stage, done: true })))
      setResult(payload)
      setAppState(payload.state)
    } catch {
      window.clearTimeout(t1)
      window.clearTimeout(t2)
      setLoadingStages((prev) => prev.map((stage) => ({ ...stage, done: true })))
      setResult({
        state: 'error',
        question: query,
        answer: 'Could not reach the ASTRION API. Start the API server and try again.',
        sources: [],
        retrieved_chunks: [],
        retrieval_count: 0,
        latency: { retrieval_seconds: 0, generation_seconds: 0, total_seconds: 0 },
      })
      setAppState('error')
    }
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void handleSubmit()
    }
  }

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [question])

  return (
    <div className="flex flex-col min-h-full">
      <div className="flex-1 max-w-3xl mx-auto w-full px-6 py-10 md:py-12">
        <div className="mb-10 fade-rise">
          <p
            className="brand-wordmark text-4xl md:text-5xl font-semibold mb-3"
            style={{ color: 'var(--color-text-primary)' }}
          >
            ASTRION
          </p>
          <h1
            className="text-xs font-semibold tracking-[0.18em] uppercase mb-3"
            style={{ color: 'var(--color-accent)' }}
          >
            Ask Your Documents
          </h1>
          <p className="text-sm max-w-xl" style={{ color: 'var(--color-text-secondary)' }}>
            Ask a question. ASTRION retrieves evidence from your FAISS index, cites the source page,
            and refuses to invent an answer when the corpus has nothing useful.
          </p>
        </div>

        <div
          className="mb-8 rounded-xl panel-lift fade-rise-delay"
          style={{
            border: '1px solid var(--color-border-strong)',
            background: 'var(--color-surface)',
          }}
        >
          <textarea
            ref={textareaRef}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents…"
            rows={3}
            className="w-full px-4 pt-4 pb-3 text-sm resize-none outline-none"
            style={{
              background: 'transparent',
              color: 'var(--color-text-primary)',
              fontFamily: 'var(--font-sans)',
              fontSize: '0.95rem',
              lineHeight: 1.65,
              borderBottom: '1px solid var(--color-border)',
              borderRadius: '14px 14px 0 0',
              maxHeight: 200,
              minHeight: 88,
            }}
          />
          <div className="flex items-center justify-between px-4 py-3">
            <span className="text-xs" style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>
              Enter to ask · Shift+Enter for newline
            </span>
            <button
              onClick={() => void handleSubmit()}
              disabled={!question.trim() || appState === 'loading'}
              className="flex items-center gap-2 px-4 py-2 text-xs font-semibold tracking-wide"
              style={{
                background:
                  question.trim() && appState !== 'loading' ? 'var(--color-accent)' : 'var(--color-border)',
                color: question.trim() && appState !== 'loading' ? '#fff' : 'var(--color-text-muted)',
                border: 'none',
                borderRadius: 'var(--radius-sm)',
                cursor: question.trim() && appState !== 'loading' ? 'pointer' : 'not-allowed',
                letterSpacing: '0.04em',
              }}
            >
              Ask ASTRION
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path
                  d="M2 6h8M7 3l3 3-3 3"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        </div>

        {appState === 'empty' && <EmptyState onSelect={(q) => void handleSubmit(q)} ready={Boolean(stats?.ready)} />}
        {appState === 'loading' && <LoadingState stages={loadingStages} question={submittedQuestion} />}
        {appState === 'answer' && result && (
          <AnswerState result={result} traceOpen={traceOpen} setTraceOpen={setTraceOpen} />
        )}
        {appState === 'no-results' && result && (
          <NoResultsState
            answer={result.answer}
            onRetry={() => {
              setAppState('empty')
              setSubmittedQuestion('')
              setResult(null)
            }}
          />
        )}
        {appState === 'error' && result && (
          <ErrorState
            answer={result.answer}
            result={result}
            onRetry={() => void handleSubmit(submittedQuestion)}
          />
        )}
      </div>

      <SystemInfoBar stats={stats} />
    </div>
  )
}

function EmptyState({ onSelect, ready }: { onSelect: (q: string) => void; ready: boolean }) {
  return (
    <div className="fade-rise">
      <div
        className="mb-8 py-12 text-center rounded-xl panel-lift"
        style={{
          border: '1px solid var(--color-border)',
          background:
            'linear-gradient(180deg, rgba(232,240,251,0.9) 0%, rgba(251,252,254,0.98) 55%)',
        }}
      >
        <div
          className="inline-flex items-center justify-center mb-5"
          style={{
            width: 52,
            height: 52,
            background: 'var(--color-accent)',
            borderRadius: 14,
            boxShadow: '0 12px 24px rgba(31,77,143,0.28)',
          }}
        >
          <svg width="22" height="22" viewBox="0 0 18 18" fill="none">
            <path
              d="M9 2L11 7H16L12 10.5L13.5 15.5L9 12.5L4.5 15.5L6 10.5L2 7H7L9 2Z"
              fill="white"
            />
          </svg>
        </div>
        <h2 className="brand-wordmark text-2xl font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
          Ask with evidence
        </h2>
        <p className="text-sm mb-1" style={{ color: 'var(--color-text-secondary)' }}>
          Every answer is tied to a retrieved chunk, source file, and page.
        </p>
        <p className="text-xs max-w-sm mx-auto" style={{ color: 'var(--color-text-muted)' }}>
          {ready
            ? 'Your index is online. Start with an example or write your own question.'
            : 'No index yet. Place PDFs in data/documents and run python -m src.ingest.'}
        </p>
      </div>

      <div className="mb-2">
        <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--color-text-muted)' }}>
          Example Questions
        </span>
      </div>
      <div className="flex flex-col gap-2">
        {EXAMPLE_QUESTIONS.map((q, i) => (
          <button
            key={q}
            onClick={() => onSelect(q)}
            className="text-left px-4 py-3 text-sm transition-colors"
            style={{
              border: '1px solid var(--color-border)',
              background: 'var(--color-surface)',
              color: 'var(--color-text-secondary)',
              borderRadius: 'var(--radius-md)',
              cursor: 'pointer',
            }}
            onMouseEnter={(e) => {
              const el = e.currentTarget
              el.style.borderColor = 'var(--color-accent)'
              el.style.color = 'var(--color-accent)'
            }}
            onMouseLeave={(e) => {
              const el = e.currentTarget
              el.style.borderColor = 'var(--color-border)'
              el.style.color = 'var(--color-text-secondary)'
            }}
          >
            <span className="mr-2 font-mono text-[0.7rem]" style={{ color: 'var(--color-text-muted)' }}>
              {String(i + 1).padStart(2, '0')}
            </span>
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

function LoadingState({ stages, question }: { stages: LoadingStage[]; question: string }) {
  return (
    <div
      className="rounded-xl p-6 panel-lift fade-rise"
      style={{ border: '1px solid var(--color-border)', background: 'var(--color-surface)' }}
    >
      <div className="mb-5">
        <div className="text-xs font-semibold tracking-widest uppercase mb-2" style={{ color: 'var(--color-text-muted)' }}>
          Processing Query
        </div>
        <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          “{question}”
        </p>
        <div className="mt-4 h-1 rounded-full overflow-hidden" style={{ background: 'var(--color-surface-2)' }}>
          <div className="h-full shimmer-bar" />
        </div>
      </div>
      <div className="flex flex-col gap-3">
        {stages.map((stage, i) => {
          const isActive = !stage.done && (i === 0 || stages[i - 1].done)
          return (
            <div key={stage.label} className="flex items-center gap-3">
              <div
                className="shrink-0 rounded-full flex items-center justify-center"
                style={{
                  width: 20,
                  height: 20,
                  border: stage.done
                    ? 'none'
                    : isActive
                      ? '1.5px solid var(--color-accent)'
                      : '1.5px solid var(--color-border)',
                  background: stage.done ? 'var(--color-accent)' : 'transparent',
                }}
              >
                {stage.done ? (
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                    <path d="M2 5l2.5 2.5L8 3" stroke="white" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : isActive ? (
                  <span
                    style={{
                      display: 'block',
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      background: 'var(--color-accent)',
                      animation: 'pulse 1.2s ease-in-out infinite',
                    }}
                  />
                ) : null}
              </div>
              <span
                className="text-sm"
                style={{
                  color: stage.done
                    ? 'var(--color-text-secondary)'
                    : isActive
                      ? 'var(--color-text-primary)'
                      : 'var(--color-text-muted)',
                  fontWeight: isActive ? 600 : 400,
                }}
              >
                {stage.label}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function AnswerState({
  result,
  traceOpen,
  setTraceOpen,
}: {
  result: AskResponse
  traceOpen: boolean
  setTraceOpen: (value: boolean) => void
}) {
  const latency = result.latency || { retrieval_seconds: 0, generation_seconds: 0, total_seconds: 0 }
  return (
    <div className="flex flex-col gap-6 fade-rise">
      <div className="flex gap-3 items-start">
        <div
          className="shrink-0 mt-0.5 rounded"
          style={{
            width: 20,
            height: 20,
            background: 'var(--color-bg)',
            border: '1px solid var(--color-border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <circle cx="5" cy="4" r="2" stroke="var(--color-text-muted)" strokeWidth="1.2" />
            <path d="M2 9c0-1.7 1.3-3 3-3s3 1.3 3 3" stroke="var(--color-text-muted)" strokeWidth="1.2" strokeLinecap="round" />
          </svg>
        </div>
        <p className="text-sm" style={{ color: 'var(--color-text-secondary)', paddingTop: 2 }}>
          {result.question}
        </p>
      </div>

      <div
        className="rounded-xl p-6 panel-lift"
        style={{ border: '1px solid var(--color-border)', background: 'var(--color-surface)' }}
      >
        <div className="flex items-center justify-between mb-4">
          <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--color-text-muted)' }}>
            Answer
          </span>
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: '#2a9d5a' }} />
            <span className="text-xs font-mono" style={{ color: 'var(--color-text-muted)' }}>
              Grounded
            </span>
          </div>
        </div>

        <div className="text-sm mb-5 whitespace-pre-wrap" style={{ color: 'var(--color-text-primary)', lineHeight: 1.8 }}>
          {result.answer}
        </div>

        <div style={{ borderTop: '1px solid var(--color-border)', paddingTop: 16 }}>
          <div className="text-xs font-semibold tracking-widest uppercase mb-3" style={{ color: 'var(--color-text-muted)' }}>
            Sources
          </div>
          <div className="flex flex-wrap gap-2">
            {result.sources.length === 0 && (
              <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                No source citations returned.
              </span>
            )}
            {result.sources.map((source, index) => (
              <span
                key={`${source.chunk_id}-${index}`}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs"
                style={{
                  border: '1px solid var(--color-border)',
                  background: 'var(--color-bg)',
                  color: 'var(--color-text-secondary)',
                  borderRadius: 'var(--radius-sm)',
                }}
              >
                <span className="font-mono font-semibold" style={{ color: 'var(--color-accent)', fontSize: '0.65rem' }}>
                  [{index + 1}]
                </span>
                {source.source} · Page {source.page}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-px overflow-hidden rounded-xl" style={{ border: '1px solid var(--color-border)' }}>
        {[
          { label: 'Retrieval', value: `${latency.retrieval_seconds?.toFixed(2) ?? '0.00'}s` },
          { label: 'Generation', value: `${latency.generation_seconds?.toFixed(2) ?? '0.00'}s` },
          { label: 'Total', value: `${latency.total_seconds?.toFixed(2) ?? '0.00'}s` },
        ].map((item) => (
          <div key={item.label} className="px-4 py-3" style={{ background: 'var(--color-surface)' }}>
            <div className="text-[0.65rem] uppercase tracking-wider mb-1" style={{ color: 'var(--color-text-muted)' }}>
              {item.label}
            </div>
            <div className="font-mono text-sm font-semibold">{item.value}</div>
          </div>
        ))}
      </div>

      <div>
        <div className="text-xs font-semibold tracking-widest uppercase mb-3" style={{ color: 'var(--color-text-muted)' }}>
          Retrieved Chunks
        </div>
        <div className="flex flex-col gap-2">
          {result.retrieved_chunks.map((chunk) => (
            <div
              key={chunk.chunk_id}
              className="rounded-xl p-4"
              style={{ border: '1px solid var(--color-border)', background: 'var(--color-surface)' }}
            >
              <div className="flex items-center justify-between gap-3 mb-2">
                <div className="text-xs font-mono" style={{ color: 'var(--color-accent)' }}>
                  {chunk.source} · p.{chunk.page} · {chunk.chunk_id}
                </div>
                <div className="text-xs font-mono" style={{ color: 'var(--color-text-muted)' }}>
                  {chunk.score.toFixed(4)}
                </div>
              </div>
              <p className="text-sm" style={{ color: 'var(--color-text-secondary)', lineHeight: 1.65 }}>
                {chunk.text}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div
        className="rounded-xl overflow-hidden"
        style={{ border: '1px solid var(--color-border)', background: 'var(--color-surface)' }}
      >
        <button
          onClick={() => setTraceOpen(!traceOpen)}
          className="w-full flex items-center justify-between px-4 py-3 text-left"
          style={{ background: 'transparent', border: 'none', cursor: 'pointer' }}
        >
          <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--color-text-muted)' }}>
            Retrieval Trace
          </span>
          <span className="text-xs font-mono" style={{ color: 'var(--color-text-muted)' }}>
            {traceOpen ? 'Hide' : 'Show'}
          </span>
        </button>
        {traceOpen && (
          <div style={{ borderTop: '1px solid var(--color-border)' }}>
            {result.retrieved_chunks.map((chunk) => (
              <div
                key={`trace-${chunk.chunk_id}`}
                className="px-4 py-3 text-xs font-mono"
                style={{ borderBottom: '1px solid var(--color-border)', color: 'var(--color-text-secondary)' }}
              >
                <div>rank {chunk.rank}</div>
                <div>source {chunk.source}</div>
                <div>page {chunk.page}</div>
                <div>chunk {chunk.chunk_id}</div>
                <div>score {chunk.score.toFixed(4)}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function NoResultsState({ answer, onRetry }: { answer: string; onRetry: () => void }) {
  return (
    <div
      className="rounded-xl p-6 fade-rise"
      style={{ border: '1px solid var(--color-border)', background: 'var(--color-warning-bg)' }}
    >
      <div className="text-xs font-semibold tracking-widest uppercase mb-2" style={{ color: 'var(--color-warning)' }}>
        Insufficient Evidence
      </div>
      <p className="text-sm mb-4" style={{ color: 'var(--color-text-primary)' }}>
        {answer}
      </p>
      <button
        onClick={onRetry}
        className="text-xs font-semibold px-3 py-2"
        style={{
          background: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-sm)',
          cursor: 'pointer',
        }}
      >
        Ask something else
      </button>
    </div>
  )
}

function ErrorState({
  answer,
  onRetry,
  result,
}: {
  answer: string
  onRetry: () => void
  result?: AskResponse | null
}) {
  const chunks = result?.retrieved_chunks || []
  return (
    <div className="fade-rise flex flex-col gap-4">
      <div
        className="rounded-xl p-6"
        style={{ border: '1px solid var(--color-border)', background: 'var(--color-error-bg)' }}
      >
        <div className="text-xs font-semibold tracking-widest uppercase mb-2" style={{ color: 'var(--color-error)' }}>
          Generation Unavailable
        </div>
        <p className="text-sm mb-4" style={{ color: 'var(--color-text-primary)' }}>
          {answer}
        </p>
        <button
          onClick={onRetry}
          className="text-xs font-semibold px-3 py-2"
          style={{
            background: 'var(--color-surface)',
            border: '1px solid var(--color-border)',
            borderRadius: 'var(--radius-sm)',
            cursor: 'pointer',
          }}
        >
          Retry
        </button>
      </div>
      {chunks.length > 0 && (
        <div
          className="rounded-xl p-4"
          style={{ border: '1px solid var(--color-border)', background: 'var(--color-surface)' }}
        >
          <div className="text-xs font-semibold tracking-widest uppercase mb-3" style={{ color: 'var(--color-text-muted)' }}>
            Retrieved Evidence
          </div>
          {chunks.map((chunk) => (
            <div key={chunk.chunk_id} className="mb-3 last:mb-0">
              <div className="text-xs font-mono mb-1" style={{ color: 'var(--color-accent)' }}>
                {chunk.source} · p.{chunk.page} · score {chunk.score.toFixed(4)}
              </div>
              <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                {chunk.text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function SystemInfoBar({ stats }: { stats: StatsResponse | null }) {
  const items = [
    { label: 'Documents', value: String(stats?.documents ?? '—') },
    { label: 'Indexed Chunks', value: String(stats?.chunks ?? '—') },
    { label: 'Embedding Model', value: stats?.embedding_model ?? '—' },
    { label: 'Vector Store', value: stats?.vector_store ?? 'FAISS' },
    { label: 'LLM', value: stats?.llm_model ?? '—' },
  ]
  return (
    <div
      className="shrink-0 px-6 py-3"
      style={{
        borderTop: '1px solid var(--color-border)',
        background: 'rgba(251,252,254,0.85)',
        backdropFilter: 'blur(8px)',
      }}
    >
      <div className="max-w-3xl mx-auto flex flex-wrap gap-x-6 gap-y-2">
        {items.map((item) => (
          <div key={item.label} className="text-[0.7rem]">
            <span style={{ color: 'var(--color-text-muted)' }}>{item.label} </span>
            <span className="font-mono" style={{ color: 'var(--color-text-secondary)' }}>
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

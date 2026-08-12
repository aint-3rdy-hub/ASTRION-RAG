import { useEffect, useState } from 'react'
import { getEvaluation } from '../lib/api'

export default function EvaluationPage() {
  const [metrics, setMetrics] = useState<{ label: string; value: string; unit: string }[]>([])
  const [rows, setRows] = useState<
    {
      question: string
      expected_source: string
      retrieved_source: string
      result: string
      latency: string | null
    }[]
  >([])
  const [failures, setFailures] = useState<
    { title: string; description: string; cause: string; mitigation: string }[]
  >([])
  const [available, setAvailable] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [notes, setNotes] = useState<string | null>(null)
  const [failuresOpen, setFailuresOpen] = useState(true)

  useEffect(() => {
    getEvaluation()
      .then((payload) => {
        setAvailable(payload.available)
        setMessage(payload.message || null)
        setMetrics(payload.metrics || [])
        setRows(payload.rows || [])
        setFailures(payload.failures || [])
        setNotes(payload.notes || null)
      })
      .catch(() => setMessage('Could not load evaluation results from the API.'))
  }, [])

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 md:py-12 fade-rise">
      <div className="mb-8">
        <h1 className="brand-wordmark text-3xl font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
          Evaluation
        </h1>
        <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          Metrics from <span className="font-mono">evaluation/report.json</span>. Groundedness and citation
          scores stay null until scored manually.
        </p>
      </div>

      {!available && (
        <div
          className="mb-8 p-4 rounded-xl text-sm"
          style={{ background: 'var(--color-warning-bg)', color: 'var(--color-warning)' }}
        >
          {message || 'No evaluation report yet.'}
        </div>
      )}

      {metrics.length > 0 && (
        <div
          className="grid grid-cols-2 md:grid-cols-5 gap-px mb-8 overflow-hidden rounded-xl"
          style={{ border: '1px solid var(--color-border)' }}
        >
          {metrics.map((metric) => (
            <div key={metric.label} className="px-5 py-4" style={{ background: 'var(--color-surface)' }}>
              <div className="text-xs mb-2" style={{ color: 'var(--color-text-muted)' }}>
                {metric.label}
              </div>
              <div className="flex items-baseline gap-0.5">
                <span className="text-2xl font-semibold font-mono" style={{ color: 'var(--color-text-primary)' }}>
                  {metric.value}
                </span>
                {metric.unit && (
                  <span className="text-sm font-mono" style={{ color: 'var(--color-text-muted)' }}>
                    {metric.unit}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mb-2">
        <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--color-text-muted)' }}>
          Evaluation Results
        </span>
      </div>
      <div
        className="rounded-xl overflow-hidden mb-8 panel-lift"
        style={{ border: '1px solid var(--color-border)', background: 'var(--color-surface)' }}
      >
        {rows.length === 0 ? (
          <div className="px-4 py-8 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            No completed evaluation rows yet. Run <span className="font-mono">python -m evaluation.evaluate</span>.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="w-full text-xs" style={{ borderCollapse: 'collapse', minWidth: 600 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--color-border)', background: 'var(--color-bg)' }}>
                  {['Question', 'Expected Source', 'Retrieved Source', 'Result', 'Latency'].map((heading) => (
                    <th
                      key={heading}
                      className="px-4 py-3 text-left"
                      style={{
                        fontSize: '0.65rem',
                        fontWeight: 600,
                        letterSpacing: '0.1em',
                        textTransform: 'uppercase',
                        color: 'var(--color-text-muted)',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr
                    key={`${row.question}-${index}`}
                    style={{
                      borderBottom: index < rows.length - 1 ? '1px solid var(--color-border)' : 'none',
                    }}
                  >
                    <td className="px-4 py-3" style={{ color: 'var(--color-text-primary)', maxWidth: 220 }}>
                      {row.question}
                    </td>
                    <td className="px-4 py-3 font-mono" style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem' }}>
                      {row.expected_source}
                    </td>
                    <td className="px-4 py-3 font-mono" style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem' }}>
                      {row.retrieved_source}
                    </td>
                    <td className="px-4 py-3">
                      <ResultBadge result={row.result} />
                    </td>
                    <td className="px-4 py-3 font-mono" style={{ color: 'var(--color-text-muted)' }}>
                      {row.latency ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div
        className="rounded-xl overflow-hidden mb-6"
        style={{ border: '1px solid var(--color-border)', background: 'var(--color-surface)' }}
      >
        <button
          onClick={() => setFailuresOpen(!failuresOpen)}
          className="w-full flex items-center justify-between px-4 py-3"
          style={{ background: 'transparent', border: 'none', cursor: 'pointer' }}
        >
          <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--color-text-muted)' }}>
            Failure Cases
          </span>
          <span className="text-xs font-mono" style={{ color: 'var(--color-text-muted)' }}>
            {failuresOpen ? 'Hide' : 'Show'}
          </span>
        </button>
        {failuresOpen && (
          <div style={{ borderTop: '1px solid var(--color-border)' }}>
            {failures.length === 0 ? (
              <div className="px-4 py-5 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                No recorded API/pipeline failures in the latest evaluation run.
              </div>
            ) : (
              failures.map((item) => (
                <div
                  key={item.title}
                  className="px-4 py-4"
                  style={{ borderBottom: '1px solid var(--color-border)' }}
                >
                  <div className="text-sm font-semibold mb-1" style={{ color: 'var(--color-text-primary)' }}>
                    {item.title}
                  </div>
                  <p className="text-xs mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                    {item.description}
                  </p>
                  <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    Cause: {item.cause}
                  </p>
                  <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    Mitigation: {item.mitigation}
                  </p>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {notes && (
        <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          {notes}
        </p>
      )}
    </div>
  )
}

function ResultBadge({ result }: { result: string }) {
  const pass = result === 'Pass'
  const fail = result === 'Fail'
  return (
    <span
      className="inline-flex px-2 py-1 rounded text-xs font-mono"
      style={{
        background: pass ? 'var(--color-success-bg)' : fail ? 'var(--color-error-bg)' : 'var(--color-bg)',
        color: pass ? 'var(--color-success)' : fail ? 'var(--color-error)' : 'var(--color-text-muted)',
      }}
    >
      {result}
    </span>
  )
}

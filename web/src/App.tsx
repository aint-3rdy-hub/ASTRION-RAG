import { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import AskPage from './pages/AskPage'
import DocumentsPage from './pages/DocumentsPage'
import EvaluationPage from './pages/EvaluationPage'
import { getStats, type StatsResponse } from './lib/api'

export type NavPage = 'ask' | 'documents' | 'evaluation' | 'retrieval' | 'index' | 'configuration'

export default function App() {
  const [activePage, setActivePage] = useState<NavPage>('ask')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [stats, setStats] = useState<StatsResponse | null>(null)

  useEffect(() => {
    getStats()
      .then(setStats)
      .catch(() => setStats(null))
  }, [activePage])

  const pageLabels: Record<NavPage, string> = {
    ask: 'Ask',
    documents: 'Documents',
    evaluation: 'Evaluation',
    retrieval: 'Retrieval',
    index: 'Index',
    configuration: 'Configuration',
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <TopBar
        pageTitle={pageLabels[activePage]}
        ready={Boolean(stats?.ready)}
        onMenuToggle={() => setSidebarOpen((open) => !open)}
      />
      <div className="flex flex-1 min-h-0 relative">
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-20 bg-black/20 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
        <Sidebar
          activePage={activePage}
          onNavigate={(page) => {
            setActivePage(page)
            setSidebarOpen(false)
          }}
          open={sidebarOpen}
          ready={Boolean(stats?.ready)}
          documentCount={stats?.documents ?? 0}
        />
        <main className="flex-1 min-w-0 overflow-y-auto">
          {stats && stats.groq_ready === false && (
            <div
              className="mx-6 mt-4 md:mx-8 px-4 py-3 text-sm rounded-lg"
              style={{
                background: 'var(--color-error-bg)',
                color: 'var(--color-error)',
                border: '1px solid var(--color-border)',
              }}
            >
              GROQ_API_KEY is not set. Ask will return extractive citations from retrieved
              chunks until you add a real key to <code>.env</code> and restart the API.
            </div>
          )}
          {activePage === 'ask' && <AskPage stats={stats} />}
          {activePage === 'documents' && <DocumentsPage />}
          {activePage === 'evaluation' && <EvaluationPage />}
          {(activePage === 'retrieval' || activePage === 'index' || activePage === 'configuration') && (
            <PlaceholderPage title={pageLabels[activePage]} stats={stats} />
          )}
        </main>
      </div>
    </div>
  )
}

function PlaceholderPage({ title, stats }: { title: string; stats: StatsResponse | null }) {
  return (
    <div className="p-8 md:p-12 fade-rise">
      <h1
        className="text-xs font-semibold tracking-widest uppercase mb-2"
        style={{ color: 'var(--color-text-muted)' }}
      >
        {title}
      </h1>
      <p className="text-sm mb-6" style={{ color: 'var(--color-text-secondary)' }}>
        Live system snapshot from the current FAISS index.
      </p>
      <div
        className="panel-lift rounded-xl p-5 max-w-xl"
        style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}
      >
        <dl className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <dt style={{ color: 'var(--color-text-muted)' }}>Documents</dt>
            <dd className="font-mono font-semibold">{stats?.documents ?? '—'}</dd>
          </div>
          <div>
            <dt style={{ color: 'var(--color-text-muted)' }}>Chunks</dt>
            <dd className="font-mono font-semibold">{stats?.chunks ?? '—'}</dd>
          </div>
          <div>
            <dt style={{ color: 'var(--color-text-muted)' }}>Embedding</dt>
            <dd className="font-mono text-xs">{stats?.embedding_model ?? '—'}</dd>
          </div>
          <div>
            <dt style={{ color: 'var(--color-text-muted)' }}>LLM</dt>
            <dd className="font-mono text-xs">{stats?.llm_model ?? '—'}</dd>
          </div>
        </dl>
      </div>
    </div>
  )
}

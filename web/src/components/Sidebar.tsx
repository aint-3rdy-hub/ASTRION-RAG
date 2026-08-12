import type { NavPage } from '../App'

interface SidebarProps {
  activePage: NavPage
  onNavigate: (page: NavPage) => void
  open: boolean
  ready: boolean
  documentCount: number
}

const mainNav: { id: NavPage; label: string }[] = [
  { id: 'ask', label: 'Ask' },
  { id: 'documents', label: 'Documents' },
  { id: 'evaluation', label: 'Evaluation' },
]

const systemNav: { id: NavPage; label: string }[] = [
  { id: 'retrieval', label: 'Retrieval' },
  { id: 'index', label: 'Index' },
  { id: 'configuration', label: 'Configuration' },
]

export default function Sidebar({
  activePage,
  onNavigate,
  open,
  ready,
  documentCount,
}: SidebarProps) {
  return (
    <aside
      className="sidebar flex flex-col overflow-y-auto shrink-0"
      data-open={open}
      style={{
        width: 220,
        borderRight: '1px solid var(--color-border)',
        background: 'rgba(251, 252, 254, 0.92)',
        backdropFilter: 'blur(10px)',
      }}
    >
      <div className="px-5 py-6" style={{ borderBottom: '1px solid var(--color-border)' }}>
        <div className="flex items-center gap-3">
          <AstrionMark />
          <div>
            <div
              className="brand-wordmark text-lg font-semibold"
              style={{ color: 'var(--color-text-primary)', lineHeight: 1 }}
            >
              ASTRION
            </div>
            <div
              className="text-[0.7rem] font-semibold tracking-[0.18em] uppercase mt-1"
              style={{ color: 'var(--color-accent)' }}
            >
              RAG
            </div>
          </div>
        </div>
        <p className="mt-3 text-xs leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>
          Citation-grounded answers from your own documents.
        </p>
      </div>

      <nav className="flex flex-col gap-0.5 px-3 pt-4">
        {mainNav.map((item) => (
          <NavItem
            key={item.id}
            label={item.label}
            active={activePage === item.id}
            onClick={() => onNavigate(item.id)}
          />
        ))}
      </nav>

      <div className="px-3 mt-4">
        <div style={{ borderTop: '1px solid var(--color-border)' }} className="pt-4">
          <div
            className="px-3 pb-2 text-xs font-semibold tracking-widest uppercase"
            style={{ color: 'var(--color-text-muted)' }}
          >
            System
          </div>
          <div className="flex flex-col gap-0.5">
            {systemNav.map((item) => (
              <NavItem
                key={item.id}
                label={item.label}
                active={activePage === item.id}
                onClick={() => onNavigate(item.id)}
                muted
              />
            ))}
          </div>
        </div>
      </div>

      <div className="mt-auto px-5 py-4" style={{ borderTop: '1px solid var(--color-border)' }}>
        <div className="flex items-center gap-2 mb-2">
          <span
            className="inline-block rounded-full"
            style={{
              width: 7,
              height: 7,
              background: ready ? '#2a9d5a' : '#d4900a',
              flexShrink: 0,
              boxShadow: ready ? '0 0 0 3px rgba(42,157,90,0.18)' : '0 0 0 3px rgba(212,144,10,0.18)',
            }}
          />
          <span className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
            {ready ? 'Index Ready' : 'Index Missing'}
          </span>
        </div>
        <div className="text-[0.7rem] font-mono" style={{ color: 'var(--color-text-muted)' }}>
          {documentCount} document{documentCount === 1 ? '' : 's'} indexed
        </div>
      </div>
    </aside>
  )
}

function NavItem({
  label,
  active,
  onClick,
  muted = false,
}: {
  label: string
  active: boolean
  onClick: () => void
  muted?: boolean
}) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left px-3 py-2 transition-colors"
      style={{
        borderRadius: 'var(--radius-md)',
        background: active ? 'var(--color-accent-subtle)' : 'transparent',
        color: active
          ? 'var(--color-accent)'
          : muted
            ? 'var(--color-text-secondary)'
            : 'var(--color-text-primary)',
        fontSize: '0.8125rem',
        fontWeight: active ? 600 : 500,
        border: 'none',
        cursor: 'pointer',
        outline: 'none',
        boxShadow: active ? 'inset 3px 0 0 var(--color-accent)' : 'none',
      }}
      onMouseEnter={(e) => {
        if (!active) (e.currentTarget as HTMLElement).style.background = 'var(--color-bg)'
      }}
      onMouseLeave={(e) => {
        if (!active) (e.currentTarget as HTMLElement).style.background = 'transparent'
      }}
    >
      {label}
    </button>
  )
}

function AstrionMark() {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
      <defs>
        <linearGradient id="astrionMark" x1="4" y1="2" x2="30" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#2B6BC4" />
          <stop offset="1" stopColor="#163A6E" />
        </linearGradient>
      </defs>
      <rect width="34" height="34" rx="10" fill="url(#astrionMark)" />
      <path
        d="M17 7L20.2 13.5H27L21.8 17.7L23.5 24.5L17 20.8L10.5 24.5L12.2 17.7L7 13.5H13.8L17 7Z"
        fill="white"
      />
      <circle cx="17" cy="16.5" r="1.6" fill="#9EC5FF" />
    </svg>
  )
}

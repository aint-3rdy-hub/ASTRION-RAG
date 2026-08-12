interface TopBarProps {
  pageTitle: string
  ready: boolean
  onMenuToggle: () => void
}

export default function TopBar({ pageTitle, ready, onMenuToggle }: TopBarProps) {
  return (
    <header
      className="flex items-center justify-between shrink-0 px-5"
      style={{
        height: 48,
        borderBottom: '1px solid var(--color-border)',
        background: 'rgba(251, 252, 254, 0.88)',
        backdropFilter: 'blur(12px)',
      }}
    >
      <div className="flex items-center gap-3">
        <button
          className="md:hidden p-1 rounded"
          onClick={onMenuToggle}
          style={{
            border: 'none',
            background: 'transparent',
            cursor: 'pointer',
            color: 'var(--color-text-secondary)',
          }}
          aria-label="Toggle navigation"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path
              d="M2 4h12M2 8h12M2 12h12"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </button>
        <div>
          <div
            className="text-[0.65rem] font-semibold tracking-[0.16em] uppercase"
            style={{ color: 'var(--color-text-muted)' }}
          >
            ASTRION Console
          </div>
          <div className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>
            {pageTitle}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div
          className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full"
          style={{
            background: ready ? 'var(--color-success-bg)' : 'var(--color-warning-bg)',
            border: `1px solid ${ready ? 'rgba(26,99,68,0.15)' : 'rgba(138,90,0,0.15)'}`,
          }}
        >
          <span
            className="inline-block rounded-full"
            style={{ width: 6, height: 6, background: ready ? '#2a9d5a' : '#d4900a' }}
          />
          <span
            className="text-xs font-medium"
            style={{
              color: ready ? 'var(--color-success)' : 'var(--color-warning)',
              fontFamily: 'var(--font-mono)',
            }}
          >
            {ready ? 'Operational' : 'Needs Index'}
          </span>
        </div>
      </div>
    </header>
  )
}

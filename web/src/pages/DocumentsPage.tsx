import { useEffect, useRef, useState } from 'react'
import { getDocuments, uploadDocuments, type DocumentRow } from '../lib/api'

const STATUS_COLORS: Record<string, { bg: string; text: string; dot: string }> = {
  Indexed: { bg: 'var(--color-success-bg)', text: 'var(--color-success)', dot: '#2a9d5a' },
  Processing: { bg: 'var(--color-warning-bg)', text: 'var(--color-warning)', dot: '#d4900a' },
  Failed: { bg: 'var(--color-error-bg)', text: 'var(--color-error)', dot: 'var(--color-error)' },
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentRow[]>([])
  const [totals, setTotals] = useState({ documents: 0, pages: 0, chunks: 0 })
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const dragDepth = useRef(0)

  const applyPayload = (payload: {
    documents: DocumentRow[]
    total_documents: number
    total_pages: number
    total_chunks: number
    ready: boolean
  }) => {
    setDocs(payload.documents)
    setTotals({
      documents: payload.total_documents,
      pages: payload.total_pages,
      chunks: payload.total_chunks,
    })
    setReady(payload.ready)
  }

  useEffect(() => {
    getDocuments()
      .then(applyPayload)
      .catch(() => setError('Could not load indexed documents from the API.'))
  }, [])

  const handleFiles = async (fileList: FileList | File[]) => {
    const files = Array.from(fileList).filter((file) => file.name.toLowerCase().endsWith('.pdf'))
    if (!files.length) {
      setError('Only PDF files can be added.')
      setNotice(null)
      return
    }

    setUploading(true)
    setError(null)
    setNotice(null)
    try {
      const payload = await uploadDocuments(files)
      applyPayload(payload)
      setNotice(payload.message)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed. Please try again.')
    } finally {
      setUploading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 md:py-12 fade-rise">
      <div className="flex items-end justify-between mb-8 flex-wrap gap-4">
        <div>
          <h1 className="brand-wordmark text-3xl font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>
            Documents
          </h1>
          <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            Add PDFs, then ask with citation-grounded answers.
          </p>
        </div>
        <div
          className="text-xs font-mono px-3 py-2"
          style={{
            background: ready ? 'var(--color-success-bg)' : 'var(--color-warning-bg)',
            color: ready ? 'var(--color-success)' : 'var(--color-warning)',
            borderRadius: 'var(--radius-sm)',
          }}
        >
          {ready ? 'Index online' : 'Index missing'}
        </div>
      </div>

      <div
        className="upload-zone mb-6"
        data-active={dragging || uploading ? 'true' : 'false'}
        data-busy={uploading ? 'true' : 'false'}
        onDragEnter={(event) => {
          event.preventDefault()
          dragDepth.current += 1
          setDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          event.preventDefault()
          dragDepth.current = Math.max(0, dragDepth.current - 1)
          if (dragDepth.current === 0) setDragging(false)
        }}
        onDrop={(event) => {
          event.preventDefault()
          dragDepth.current = 0
          setDragging(false)
          if (!uploading) void handleFiles(event.dataTransfer.files)
        }}
        onClick={() => {
          if (!uploading) inputRef.current?.click()
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if ((event.key === 'Enter' || event.key === ' ') && !uploading) {
            event.preventDefault()
            inputRef.current?.click()
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={(event) => {
            if (event.target.files) void handleFiles(event.target.files)
          }}
        />
        <div className="upload-zone-copy">
          <div className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>
            {uploading ? 'Indexing…' : dragging ? 'Drop to add' : 'Drop PDFs here'}
          </div>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-secondary)' }}>
            {uploading
              ? 'Embedding and rebuilding the local index.'
              : 'or click to browse — PDF only, up to 25 MB each'}
          </p>
        </div>
        {uploading && <div className="upload-zone-shimmer" aria-hidden />}
      </div>

      {error && (
        <div className="mb-4 text-sm" style={{ color: 'var(--color-error)' }}>
          {error}
        </div>
      )}
      {notice && (
        <div className="mb-4 text-sm fade-rise" style={{ color: 'var(--color-success)' }}>
          {notice}
        </div>
      )}

      <div
        className="grid grid-cols-3 gap-px mb-6 overflow-hidden"
        style={{ border: '1px solid var(--color-border)', borderRadius: 'var(--radius-lg)' }}
      >
        {[
          { label: 'Documents', value: totals.documents },
          { label: 'Total Pages', value: totals.pages },
          { label: 'Indexed Chunks', value: totals.chunks },
        ].map((item) => (
          <div key={item.label} className="px-5 py-4" style={{ background: 'var(--color-surface)' }}>
            <div className="text-xs mb-1 font-mono" style={{ color: 'var(--color-text-muted)' }}>
              {item.label}
            </div>
            <div className="text-xl font-semibold font-mono" style={{ color: 'var(--color-text-primary)' }}>
              {item.value}
            </div>
          </div>
        ))}
      </div>

      <div
        className="overflow-hidden panel-lift"
        style={{
          border: '1px solid var(--color-border)',
          background: 'var(--color-surface)',
          borderRadius: 'var(--radius-lg)',
        }}
      >
        {docs.length === 0 ? (
          <div className="px-5 py-10 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            No documents yet. Drop a PDF above to build the index.
          </div>
        ) : (
          <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--color-border)', background: 'var(--color-bg)' }}>
                {['Document', 'Pages', 'Chunks', 'Status'].map((heading) => (
                  <th
                    key={heading}
                    className="px-5 py-3 text-left"
                    style={{
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      letterSpacing: '0.1em',
                      textTransform: 'uppercase',
                      color: 'var(--color-text-muted)',
                    }}
                  >
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {docs.map((doc, index) => {
                const style = STATUS_COLORS[doc.status] || STATUS_COLORS.Indexed
                return (
                  <tr
                    key={doc.id}
                    style={{
                      borderBottom: index < docs.length - 1 ? '1px solid var(--color-border)' : 'none',
                    }}
                  >
                    <td className="px-5 py-3.5">
                      <span className="font-medium text-xs font-mono" style={{ color: 'var(--color-text-primary)' }}>
                        {doc.filename}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 text-xs font-mono" style={{ color: 'var(--color-text-secondary)' }}>
                      {doc.pages}
                    </td>
                    <td className="px-5 py-3.5 text-xs font-mono" style={{ color: 'var(--color-text-secondary)' }}>
                      {doc.chunks}
                    </td>
                    <td className="px-5 py-3.5">
                      <span
                        className="inline-flex items-center gap-1.5 px-2 py-1 text-xs font-medium font-mono"
                        style={{ background: style.bg, color: style.text, borderRadius: 3 }}
                      >
                        <span
                          className="inline-block rounded-full"
                          style={{ width: 5, height: 5, background: style.dot }}
                        />
                        {doc.status}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

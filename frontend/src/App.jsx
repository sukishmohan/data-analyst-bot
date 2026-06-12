import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_BASE = '/api'

function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [profile, setProfile] = useState(null)
  const [result, setResult] = useState(null)
  const [sampleQueries, setSampleQueries] = useState([])
  const [error, setError] = useState(null)
  const [llmOk, setLlmOk] = useState(false)
  const [datasetName, setDatasetName] = useState(null)
  const [datasetLoaded, setDatasetLoaded] = useState(false)
  const fileInputRef = useRef(null)
  const resultRef = useRef(null)

  useEffect(() => {
    fetchHealth()
    fetchSampleQueries()
  }, [])

  async function fetchHealth() {
    try {
      const r = await fetch(`${API_BASE}/health`)
      const d = await r.json()
      setLlmOk(d.llm_configured)
      setDatasetName(d.dataset_name)
      setDatasetLoaded(d.dataset_loaded)
      if (d.dataset_loaded) fetchProfile()
    } catch {
      setLlmOk(false)
    }
  }

  async function fetchProfile() {
    try {
      const r = await fetch(`${API_BASE}/profile`)
      const d = await r.json()
      setProfile(d)
      setDatasetName(d.dataset_name)
      setDatasetLoaded(true)
    } catch {
      setProfile(null)
      setDatasetLoaded(false)
    }
  }

  async function fetchSampleQueries() {
    try {
      const r = await fetch(`${API_BASE}/sample-queries`)
      setSampleQueries(await r.json())
    } catch {}
  }

  async function handleUpload(file) {
    if (!file || !file.name.endsWith('.csv')) {
      setError('Please select a .csv file.')
      return
    }

    setUploading(true)
    setError(null)
    setResult(null)
    setProfile(null)

    const form = new FormData()
    form.append('file', file)

    try {
      const r = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form })
      if (!r.ok) {
        const err = await r.json()
        throw new Error(err.detail || 'Upload failed')
      }
      await fetchProfile()
      setDatasetLoaded(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0]
    if (file) handleUpload(file)
  }

  function handleDrop(e) {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) handleUpload(file)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!query.trim() || loading) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim() }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Analysis failed')
      }
      const data = await res.json()
      setResult(data)
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth' }), 100)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function handleSampleClick(q) {
    setQuery(q)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <span className="logo">🤖</span>
            <div>
              <h1>AI Data Analyst</h1>
              <p className="header-sub">Powered by NVIDIA NIM</p>
            </div>
          </div>
          <div className="header-right">
            <span className={`status-dot ${llmOk ? 'online' : 'offline'}`} />
            <span className="status-text">{llmOk ? 'API Ready' : 'No API Key'}</span>
          </div>
        </div>
      </header>

      <div className="layout">
        <aside className="sidebar">
          {!llmOk && (
            <div className="sidebar-warning">
              ⚠️ Set <code>NVIDIA_API_KEY</code> environment variable
            </div>
          )}

          <div className="sidebar-section">
            <h3>Upload Dataset</h3>
            <div
              className={`upload-zone ${uploading ? 'uploading' : ''}`}
              onDragOver={e => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                hidden
                onChange={handleFileChange}
              />
              {uploading ? (
                <div className="upload-spinner" />
              ) : (
                <>
                  <span className="upload-icon">📂</span>
                  <span className="upload-text">
                    {datasetLoaded ? 'Click to replace' : 'Drop CSV here or click'}
                  </span>
                </>
              )}
            </div>
          </div>

          {datasetLoaded && profile && (
            <>
              <div className="sidebar-section">
                <h3>Dataset</h3>
                <div className="dataset-name">{datasetName}</div>
                <div className="stat-grid">
                  <div className="stat">
                    <span className="stat-value">{profile.shape.rows.toLocaleString()}</span>
                    <span className="stat-label">Rows</span>
                  </div>
                  <div className="stat">
                    <span className="stat-value">{profile.shape.columns}</span>
                    <span className="stat-label">Columns</span>
                  </div>
                  <div className="stat">
                    <span className="stat-value">{profile.numeric_columns.length}</span>
                    <span className="stat-label">Numeric</span>
                  </div>
                  <div className="stat">
                    <span className="stat-value">{profile.categorical_columns.length}</span>
                    <span className="stat-label">Categorical</span>
                  </div>
                </div>
              </div>

              <div className="sidebar-section">
                <h3>Columns</h3>
                <div className="column-list">
                  {profile.columns.map(col => (
                    <div key={col.name} className="column-item">
                      <span className="column-name">{col.name}</span>
                      <span className={`column-type type-${col.type.startsWith('float') || col.type.startsWith('int') ? 'num' : 'cat'}`}>
                        {col.type}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {!datasetLoaded && !uploading && (
            <div className="sidebar-section">
              <p className="text-muted">Upload a CSV to get started</p>
            </div>
          )}
        </aside>

        <main className="main">
          {!datasetLoaded ? (
            <div className="welcome">
              <div className="welcome-icon">📊</div>
              <h2>Welcome to AI Data Analyst</h2>
              <p>Upload a CSV file to start exploring your data with natural language queries.</p>
            </div>
          ) : (
            <>
              <form className="query-form" onSubmit={handleSubmit}>
                <div className="input-row">
                  <input
                    className="query-input"
                    type="text"
                    placeholder="Ask a question about your data..."
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    disabled={loading}
                  />
                  <button className="submit-btn" type="submit" disabled={loading || !query.trim()}>
                    {loading ? (
                      <span className="spinner" />
                    ) : (
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13" />
                        <polygon points="22 2 15 22 11 13 2 9 22 2" />
                      </svg>
                    )}
                  </button>
                </div>
              </form>

              {datasetLoaded && sampleQueries.length > 0 && !result && (
                <div className="samples-row">
                  {sampleQueries.map(q => (
                    <button key={q} className="sample-btn" onClick={() => handleSampleClick(q)} disabled={loading}>
                      {q}
                    </button>
                  ))}
                </div>
              )}

              {error && (
                <div className="error-box">
                  <strong>Error:</strong> {error}
                </div>
              )}

              {loading && (
                <div className="loading-box">
                  <div className="loading-steps">
                    <div className="loading-step active">🧠 Understanding</div>
                    <div className="loading-step">📋 Planning</div>
                    <div className="loading-step">💻 Coding</div>
                    <div className="loading-step">⚙️ Executing</div>
                    <div className="loading-step">💡 Generating insights</div>
                  </div>
                </div>
              )}

              {result && (
                <div className="result" ref={resultRef}>
                  <div className="result-header">
                    <h2>{result.title}</h2>
                    <span className="confidence-badge">
                      Confidence: {Math.round(result.confidence * 100)}%
                    </span>
                  </div>

                  {result.steps.length > 0 && (
                    <div className="section">
                      <h3>Execution Plan</h3>
                      <div className="steps">
                        {result.steps.map((s, i) => (
                          <div key={i} className="step">
                            <span className="step-num">{s.step_number || i + 1}</span>
                            <span className="step-type">{s.type}</span>
                            <span className="step-action">{s.action}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {result.result_text && (
                    <div className="section">
                      <h3>Results</h3>
                      <pre className="result-text">{result.result_text}</pre>
                    </div>
                  )}

                  {result.chart_url && (
                    <div className="section">
                      <h3>Chart</h3>
                      <div className="chart-container">
                        <img src={result.chart_url} alt={result.title} />
                      </div>
                    </div>
                  )}

                  {result.insights && (
                    <div className="section">
                      <h3>Insights</h3>
                      <div className="insights">
                        {result.insights.split('\n').filter(l => l.trim()).map((line, i) => (
                          <p key={i} className="insight-line">{line}</p>
                        ))}
                      </div>
                    </div>
                  )}

                  <details className="code-details">
                    <summary>View Generated Code</summary>
                    <pre className="code-block">{result.code}</pre>
                  </details>

                  <div className="report-bar">
                    <a className="report-btn" href={`${API_BASE}/report/${result.id}`} target="_blank">
                      📄 Download PDF Report
                    </a>
                  </div>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  )
}

export default App

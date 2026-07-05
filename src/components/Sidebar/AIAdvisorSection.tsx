import { useState, useRef, useEffect } from 'react';
import { useSimulationStore } from '../../stores/simulationStore';
import { askAIAdvisor } from '../../api/client';
import type { AIAdvisorResponse } from '../../api/client';

const SUGGESTED = [
  'Best route from SoFi to LAX after a game?',
  'Reduce downtown congestion at opening ceremony?',
  'Avoid areas for pedestrian routing at night?',
  'Spread traffic between SoFi and Rose Bowl?',
];

export function AIAdvisorSection() {
  const [query, setQuery]     = useState('');
  const [model, setModel]     = useState('llama3.2');
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed]  = useState(0);
  const timerRef               = useRef<ReturnType<typeof setInterval> | null>(null);
  const [result, setResult]   = useState<AIAdvisorResponse | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const textareaRef           = useRef<HTMLTextAreaElement>(null);

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  const globalIntensity = useSimulationStore((s) => s.globalIntensity);
  const mode            = useSimulationStore((s) => s.mode);
  const timeOfDay       = useSimulationStore((s) => s.timeOfDay);
  const mlPredictions   = useSimulationStore((s) => s.mlPredictions);

  const handleAsk = async () => {
    const q = query.trim();
    if (!q || loading) return;
    setLoading(true);
    setElapsed(0);
    setResult(null);
    setError(null);
    timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    try {
      const res = await askAIAdvisor({
        query: q,
        model: model.trim() || 'llama3.2',
        simulation_context: {
          globalIntensity,
          mode,
          timeOfDay,
          ...(mlPredictions && {
            mlPredictions: {
              surgeIntensity: mlPredictions.surgeIntensity,
              confidence: mlPredictions.confidence,
              condition: mlPredictions.condition,
            },
          }),
        },
      });
      setResult(res);
    } catch {
      setError('Could not reach the backend. Make sure the FastAPI server is running.');
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
      setLoading(false);
    }
  };

  const handleSuggestion = (s: string) => {
    setQuery(s);
    textareaRef.current?.focus();
  };

  return (
    <div>
      <div
        className="flex items-center gap-2 mb-2"
        style={{ borderBottom: '1px solid #1E3A4A', paddingBottom: '6px' }}
      >
        <span style={{ fontSize: '9px', fontWeight: 500, color: '#4B5563', letterSpacing: '0.15em', textTransform: 'uppercase', fontFamily: "'IBM Plex Sans Condensed', sans-serif" }}>
          AI Traffic Advisor
        </span>
      </div>

      <div style={{ border: '1px solid #1E3A4A', borderRadius: '4px', overflow: 'hidden', background: '#080E14' }}>
        {/* System status header */}
        <div
          className="px-3 py-2 flex items-center justify-between"
          style={{ borderBottom: '1px solid #1E3A4A', background: 'rgba(26,122,74,0.05)' }}
        >
          <div className="flex items-center gap-2">
            <div
              className="animate-status-blink"
              style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#1A7A4A', flexShrink: 0 }}
            />
            <span style={{ fontSize: '9px', fontWeight: 700, color: '#1A7A4A', letterSpacing: '0.12em', fontFamily: "'IBM Plex Sans Condensed', sans-serif" }}>
              ADVISORY SYSTEM ONLINE
            </span>
          </div>
          <span style={{ fontSize: '8px', color: '#4B5563', fontFamily: "'IBM Plex Mono', monospace", letterSpacing: '0.04em' }}>
            OlympiFlow-RAG v1.0
          </span>
        </div>

        <div className="p-3 space-y-3">
          {/* Model input */}
          <div className="flex items-center gap-2">
            <span style={{ fontSize: '9px', color: '#4B5563', letterSpacing: '0.08em', fontFamily: "'IBM Plex Sans Condensed', sans-serif', width: '36px', flexShrink: 0" }}>
              MODEL
            </span>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="llama3.2"
              className="flex-1 outline-none transition-colors"
              style={{
                background: '#0F1923',
                border: '1px solid #1E3A4A',
                borderRadius: '3px',
                padding: '4px 8px',
                fontSize: '10px',
                color: '#C9D1D9',
                fontFamily: "'IBM Plex Mono', monospace",
              }}
              onFocus={(e) => { e.currentTarget.style.borderColor = '#0066CC50'; }}
              onBlur={(e) => { e.currentTarget.style.borderColor = '#1E3A4A'; }}
            />
          </div>

          {/* Suggested queries as chips */}
          <div>
            <div style={{ fontSize: '8px', color: '#4B5563', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: '6px', fontFamily: "'IBM Plex Sans Condensed', sans-serif" }}>
              SUGGESTED QUERIES
            </div>
            <div className="flex flex-wrap gap-1">
              {SUGGESTED.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSuggestion(s)}
                  style={{
                    fontSize: '8px',
                    padding: '3px 7px',
                    borderRadius: '3px',
                    border: '1px solid #1E3A4A',
                    color: '#5C7A8A',
                    background: 'transparent',
                    fontFamily: "'IBM Plex Sans Condensed', sans-serif",
                    lineHeight: 1.4,
                    textAlign: 'left',
                    cursor: 'pointer',
                    transition: 'border-color 0.15s, color 0.15s',
                  }}
                  onMouseEnter={(e) => {
                    const el = e.currentTarget as HTMLElement;
                    el.style.borderColor = '#0066CC50';
                    el.style.color = '#0066CC';
                  }}
                  onMouseLeave={(e) => {
                    const el = e.currentTarget as HTMLElement;
                    el.style.borderColor = '#1E3A4A';
                    el.style.color = '#5C7A8A';
                  }}
                >
                  {s.length > 40 ? s.slice(0, 40) + '…' : s}
                </button>
              ))}
            </div>
          </div>

          {/* Query textarea */}
          <textarea
            ref={textareaRef}
            rows={3}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleAsk(); }}
            placeholder="Ask about traffic routing, congestion, or Olympic logistics..."
            className="w-full outline-none resize-none"
            style={{
              background: '#0F1923',
              border: '1px solid #1E3A4A',
              borderRadius: '3px',
              padding: '8px 10px',
              fontSize: '10px',
              color: '#C9D1D9',
              fontFamily: "'IBM Plex Sans', sans-serif",
              lineHeight: 1.5,
            }}
            onFocus={(e) => { e.currentTarget.style.borderColor = '#0066CC50'; }}
            onBlur={(e) => { e.currentTarget.style.borderColor = '#1E3A4A'; }}
          />

          {/* Submit */}
          <button
            onClick={handleAsk}
            disabled={!query.trim() || loading}
            className="w-full flex items-center justify-center gap-2 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
            style={{
              border: '1px solid #0066CC40',
              color: '#0066CC',
              fontSize: '10px',
              fontWeight: 600,
              letterSpacing: '0.08em',
              fontFamily: "'IBM Plex Sans Condensed', sans-serif",
              background: 'transparent',
              borderRadius: '3px',
              height: '32px',
            }}
            onMouseEnter={(e) => { if (query.trim() && !loading) (e.currentTarget as HTMLElement).style.background = 'rgba(0,102,204,0.10)'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
          >
            {loading ? (
              <>
                <SpinnerIcon />
                PROCESSING… {elapsed}s
              </>
            ) : (
              <>
                <BrainIcon />
                ASK ADVISOR
                <span style={{ fontSize: '8px', color: '#4B5563', fontWeight: 400, marginLeft: 'auto' }}>⌘↵</span>
              </>
            )}
          </button>

          {/* Error */}
          {error && (
            <div style={{
              fontSize: '9px',
              color: '#C0392B',
              background: 'rgba(192,57,43,0.08)',
              border: '1px solid rgba(192,57,43,0.2)',
              borderRadius: '3px',
              padding: '8px 10px',
              lineHeight: 1.5,
              fontFamily: "'IBM Plex Sans', sans-serif",
            }}>
              {error}
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="space-y-2">
              <div className="flex flex-wrap gap-1">
                {result.context_used.map((ctx) => (
                  <span
                    key={ctx}
                    style={{
                      fontSize: '8px',
                      padding: '2px 6px',
                      borderRadius: '2px',
                      border: '1px solid #1E3A4A',
                      color: '#4B5563',
                      fontFamily: "'IBM Plex Sans Condensed', sans-serif",
                      letterSpacing: '0.06em',
                    }}
                  >
                    {ctx}
                  </span>
                ))}
              </div>

              <div style={{
                background: '#0F1923',
                border: '1px solid #1E3A4A',
                borderRadius: '3px',
                padding: '10px 12px',
                fontSize: '10px',
                color: '#C9D1D9',
                lineHeight: 1.6,
                whiteSpace: 'pre-wrap',
                maxHeight: '200px',
                overflowY: 'auto',
                fontFamily: "'IBM Plex Sans', sans-serif",
              }}>
                {result.answer}
              </div>

              <div style={{ fontSize: '8px', color: '#4B5563', textAlign: 'right', fontFamily: "'IBM Plex Mono', monospace" }}>
                via {result.model}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SpinnerIcon() {
  return (
    <svg
      className="animate-spin"
      width="11" height="11" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5"
      style={{ animationDuration: '0.7s' }}
    >
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>
    </svg>
  );
}

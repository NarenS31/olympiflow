import { useState, useRef } from 'react';
import { useSimulationStore } from '../../stores/simulationStore';
import { askAIAdvisor } from '../../api/client';
import type { AIAdvisorResponse } from '../../api/client';

const SUGGESTED = [
  'Best route from SoFi Stadium to LAX after a game?',
  'How to reduce downtown congestion during opening ceremony?',
  'Which areas should we avoid for pedestrian routing at night?',
  'How to spread traffic between SoFi and Rose Bowl on the same day?',
];

export function AIAdvisorSection() {
  const [query, setQuery]     = useState('');
  const [model, setModel]     = useState('llama3.2');
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState<AIAdvisorResponse | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const textareaRef           = useRef<HTMLTextAreaElement>(null);

  const globalIntensity = useSimulationStore((s) => s.globalIntensity);
  const mode            = useSimulationStore((s) => s.mode);
  const timeOfDay       = useSimulationStore((s) => s.timeOfDay);

  const handleAsk = async () => {
    const q = query.trim();
    if (!q || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const res = await askAIAdvisor({
        query: q,
        model: model.trim() || 'llama3.2',
        simulation_context: { globalIntensity, mode, timeOfDay },
      });
      setResult(res);
    } catch {
      setError('Could not reach the backend. Make sure the FastAPI server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestion = (s: string) => {
    setQuery(s);
    textareaRef.current?.focus();
  };

  return (
    <div>
      <div className="text-[10px] font-semibold text-surface-600 uppercase tracking-widest mb-2">
        AI Traffic Advisor
      </div>

      <div className="rounded-xl border border-surface-700/40 bg-surface-800/20 overflow-hidden">
        {/* Header strip */}
        <div className="px-3 py-2 border-b border-surface-700/30 flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-violet-500 flex-shrink-0" />
          <span className="text-[10px] text-surface-500 flex-1">Powered by Ollama (local LLM + RAG)</span>
        </div>

        <div className="p-3 space-y-2.5">
          {/* Model input */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-surface-500 flex-shrink-0 w-10">Model</span>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="llama3.2"
              className="flex-1 bg-surface-900/60 border border-surface-700/40 rounded-md px-2 py-1 text-[10px] text-slate-300 placeholder-surface-600 outline-none focus:border-violet-800/60 transition-colors font-mono"
            />
          </div>

          {/* Suggested questions */}
          <div className="space-y-1">
            <div className="text-[9px] text-surface-600 uppercase tracking-wider">Suggested</div>
            <div className="flex flex-wrap gap-1">
              {SUGGESTED.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSuggestion(s)}
                  className="text-[9px] px-2 py-1 rounded-full border border-surface-700/40 text-surface-500 hover:border-violet-800/50 hover:text-violet-400 transition-colors leading-tight text-left"
                >
                  {s.length > 42 ? s.slice(0, 42) + '…' : s}
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
            className="w-full bg-surface-900/60 border border-surface-700/40 rounded-lg px-2.5 py-2 text-[11px] text-slate-300 placeholder-surface-600 outline-none focus:border-violet-800/60 transition-colors resize-none leading-relaxed"
          />

          {/* Submit */}
          <button
            onClick={handleAsk}
            disabled={!query.trim() || loading}
            className="w-full py-2 rounded-lg text-[11px] font-semibold border transition-all disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            style={{ borderColor: '#4c1d95', color: '#a78bfa' }}
            onMouseEnter={(e) => { if (query.trim() && !loading) (e.currentTarget as HTMLElement).style.background = 'rgba(109,40,217,0.12)'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
          >
            {loading ? (
              <>
                <SpinnerIcon />
                Thinking…
              </>
            ) : (
              <>
                <BrainIcon />
                Ask Advisor
                <span className="text-[9px] text-surface-600 font-normal ml-auto">⌘↵</span>
              </>
            )}
          </button>

          {/* Error */}
          {error && (
            <div className="text-[10px] text-red-500/80 bg-red-950/20 border border-red-900/30 rounded-lg px-3 py-2 leading-relaxed">
              {error}
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="space-y-2">
              {/* Context chips */}
              <div className="flex flex-wrap gap-1">
                {result.context_used.map((ctx) => (
                  <span
                    key={ctx}
                    className="text-[8px] px-1.5 py-0.5 rounded-full border border-violet-900/40 text-violet-500/70 bg-violet-950/20"
                  >
                    {ctx}
                  </span>
                ))}
              </div>

              {/* Answer */}
              <div className="bg-surface-900/70 border border-surface-700/30 rounded-lg px-3 py-2.5 text-[11px] text-slate-300 leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto">
                {result.answer}
              </div>

              <div className="text-[9px] text-surface-600 text-right">
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
      width="12" height="12" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5"
      style={{ animationDuration: '0.7s' }}
    >
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>
    </svg>
  );
}

import { useSimulationStore } from '../../stores/simulationStore';
import type { SimulationMode } from '../../types';

const MODES: { id: SimulationMode; label: string; color: string }[] = [
  { id: 'baseline', label: 'Baseline',   color: '#64748b' },
  { id: 'event',    label: 'Event Day',  color: '#92400e' },
  { id: 'crisis',   label: 'Crisis',     color: '#991b1b' },
];

const OLYMPIC_COLORS = ['#0081C8', '#FCB131', '#EE334E', '#00A651', '#c8c8c8'];

export function Header() {
  const mode = useSimulationStore((s) => s.mode);
  const setMode = useSimulationStore((s) => s.setMode);
  const metrics = useSimulationStore((s) => s.metrics);

  const delayColor = metrics.avgDelayIncrease > 40 ? 'text-red-500' : metrics.avgDelayIncrease > 15 ? 'text-amber-600' : 'text-slate-400';
  const congColor  = metrics.congestionScore > 0.7  ? 'text-red-500' : metrics.congestionScore > 0.4  ? 'text-amber-600' : 'text-slate-400';

  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 h-11 flex items-stretch border-b"
      style={{ background: 'rgba(5,8,14,0.97)', borderColor: '#141f2e', backdropFilter: 'blur(16px)' }}
    >
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-5 border-r" style={{ borderColor: '#141f2e' }}>
        <div className="flex gap-[3px] items-center">
          {OLYMPIC_COLORS.map((c, i) => (
            <div key={i} className="w-[5px] h-[5px] rounded-full" style={{ background: c, opacity: 0.65 }} />
          ))}
        </div>
        <span className="text-[13px] font-semibold text-slate-200 tracking-tight">OlympiFlow</span>
        <span className="text-[9px] font-mono text-slate-700 uppercase tracking-[0.12em] hidden sm:block">
          LA&nbsp;2028&nbsp;·&nbsp;Transport&nbsp;Model
        </span>
      </div>

      {/* Mode tabs — underline style */}
      <div className="flex items-stretch px-1">
        {MODES.map((m) => {
          const active = mode === m.id;
          return (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className="relative flex items-center px-4 text-[11px] font-medium tracking-wide transition-colors duration-150"
              style={{ color: active ? m.color : '#475569' }}
            >
              {m.label}
              {active && (
                <span
                  className="absolute bottom-0 left-3 right-3 h-[2px] rounded-t-full"
                  style={{ background: m.color }}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Inline stats */}
      <div className="flex items-stretch divide-x" style={{ borderColor: '#141f2e' }}>
        <div className="flex items-center gap-2 px-4" style={{ borderColor: '#141f2e' }}>
          <span className="text-[9px] font-mono text-slate-700 uppercase tracking-widest">Cong</span>
          <span className={`text-[12px] font-mono font-semibold tabular-nums ${congColor}`}>
            {Math.round(metrics.congestionScore * 100)}%
          </span>
        </div>
        <div className="flex items-center gap-2 px-4">
          <span className="text-[9px] font-mono text-slate-700 uppercase tracking-widest">Delay</span>
          <span className={`text-[12px] font-mono font-semibold tabular-nums ${delayColor}`}>
            +{metrics.avgDelayIncrease}%
          </span>
        </div>
        {metrics.estimatedPersonsAffected > 0 && (
          <div className="flex items-center gap-2 px-4">
            <span className="text-[9px] font-mono text-slate-700 uppercase tracking-widest">Pop</span>
            <span className="text-[12px] font-mono font-semibold tabular-nums text-slate-400">
              {(metrics.estimatedPersonsAffected / 1000).toFixed(0)}K
            </span>
          </div>
        )}
        <div className="flex items-center gap-2 px-4">
          <div className="w-[5px] h-[5px] rounded-full bg-green-800" />
          <span className="text-[9px] font-mono text-slate-700 tracking-widest uppercase">Live</span>
        </div>
      </div>
    </header>
  );
}

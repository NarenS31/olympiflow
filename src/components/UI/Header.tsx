import { useSimulationStore } from '../../stores/simulationStore';
import type { SimulationMode } from '../../types';

const MODES: { id: SimulationMode; label: string }[] = [
  { id: 'baseline', label: 'Baseline'  },
  { id: 'event',    label: 'Event Day' },
  { id: 'crisis',   label: 'Crisis'    },
];

const RINGS = ['#0081C8', '#FCB131', '#EE334E', '#00A651', '#C8C8C8'];

export function Header() {
  const mode    = useSimulationStore((s) => s.mode);
  const setMode = useSimulationStore((s) => s.setMode);
  const metrics = useSimulationStore((s) => s.metrics);

  const congPct  = Math.round(metrics.congestionScore * 100);
  const delayPct = metrics.avgDelayIncrease;

  const congColor  = congPct  > 70 ? '#FF453A' : congPct  > 40 ? '#FF9F0A' : '#30D158';
  const delayColor = delayPct > 40 ? '#FF453A' : delayPct > 15 ? '#FF9F0A' : '#30D158';

  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 flex items-center"
      style={{
        height: '44px',
        background: '#1C1C1E',
        borderBottom: '0.5px solid rgba(255,255,255,0.08)',
      }}
    >
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-4 flex-shrink-0" style={{ minWidth: '196px' }}>
        <div className="flex items-center">
          {RINGS.map((c, i) => (
            <div
              key={i}
              className="rounded-full flex-shrink-0"
              style={{
                width: '9px', height: '9px',
                background: c,
                marginLeft: i > 0 ? '-4px' : 0,
                opacity: 0.9,
              }}
            />
          ))}
        </div>
        <div>
          <div style={{
            fontSize: '14px', fontWeight: 600,
            color: 'rgba(255,255,255,0.9)',
            fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
            letterSpacing: '-0.02em', lineHeight: 1,
          }}>
            OlympiFlow
          </div>
          <div style={{
            fontSize: '9px', color: 'rgba(255,255,255,0.3)',
            fontWeight: 500, letterSpacing: '0.08em',
            textTransform: 'uppercase', lineHeight: 1, marginTop: '2px',
          }}>
            LA28 Transport
          </div>
        </div>
      </div>

      {/* Separator */}
      <div style={{ width: '0.5px', height: '20px', background: 'rgba(255,255,255,0.08)', flexShrink: 0 }} />

      {/* Mode tabs */}
      <div className="flex items-center h-full">
        {MODES.map((m) => {
          const active = mode === m.id;
          return (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className="relative h-full flex items-center px-4 transition-colors"
              style={{ background: 'transparent', border: 'none', cursor: 'pointer' }}
            >
              <span style={{
                fontSize: '13px',
                fontWeight: 500,
                color: active ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.35)',
                transition: 'color 0.15s',
                letterSpacing: '-0.01em',
              }}>
                {m.label}
              </span>
              {active && (
                <div
                  className="absolute bottom-0 left-0 right-0"
                  style={{ height: '2px', background: '#0A84FF' }}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Stat chips */}
      <div className="flex items-center gap-2 px-4">
        <StatChip label="Congestion" value={`${congPct}%`} color={congColor} />
        <StatChip label="Delay"      value={`+${delayPct}%`} color={delayColor} />
        <StatChip label="Venues"     value="12 / 49" color="rgba(255,255,255,0.6)" />
      </div>

      {/* System status */}
      <div
        className="flex items-center gap-3 px-4 flex-shrink-0"
        style={{ borderLeft: '0.5px solid rgba(255,255,255,0.08)' }}
      >
        <div className="flex items-center gap-1.5">
          <div
            className="rounded-full animate-status-blink"
            style={{ width: '6px', height: '6px', background: '#30D158' }}
          />
          <span style={{
            fontSize: '11px', color: 'rgba(255,255,255,0.3)',
            letterSpacing: '0.04em', fontWeight: 500,
          }}>
            Live
          </span>
        </div>
        <button
          style={{
            fontSize: '13px', color: '#0A84FF',
            background: 'none', border: 'none',
            cursor: 'pointer', fontWeight: 400,
            fontFamily: 'inherit',
            letterSpacing: '-0.01em',
          }}
        >
          Export
        </button>
        <button
          style={{
            fontSize: '13px', color: '#0A84FF',
            background: 'none', border: 'none',
            cursor: 'pointer', fontWeight: 400,
            fontFamily: 'inherit',
            letterSpacing: '-0.01em',
          }}
        >
          Scenario
        </button>
      </div>
    </header>
  );
}

function StatChip({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div
      className="flex items-center gap-1.5 glass-surface"
      style={{
        borderRadius: '8px',
        padding: '4px 10px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.15)',
      }}
    >
      <span style={{
        fontSize: '11px',
        color: 'rgba(255,255,255,0.35)',
        fontWeight: 400,
        letterSpacing: '-0.01em',
      }}>
        {label}
      </span>
      <span style={{
        fontSize: '12px',
        fontWeight: 600,
        color,
        fontFamily: "'SF Mono', ui-monospace, monospace",
        letterSpacing: '-0.02em',
      }}>
        {value}
      </span>
    </div>
  );
}

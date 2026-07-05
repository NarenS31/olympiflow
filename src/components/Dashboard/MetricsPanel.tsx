import { useSimulationStore } from '../../stores/simulationStore';
import { CongestionChart } from './CongestionChart';

const CONDITION_COLORS: Record<string, string> = {
  NORMAL:   '#30D158',
  ELEVATED: '#FF9F0A',
  PEAK:     '#FF6B2B',
  CRITICAL: '#FF453A',
};

export function MetricsPanel() {
  const metrics           = useSimulationStore((s) => s.metrics);
  const mode              = useSimulationStore((s) => s.mode);
  const staggeredArrivals = useSimulationStore((s) => s.staggeredArrivals);
  const globalIntensity   = useSimulationStore((s) => s.globalIntensity);
  const mlPredictions     = useSimulationStore((s) => s.mlPredictions);

  const modeLabel = mode === 'crisis'   ? 'Crisis Scenario'
                  : mode === 'event'    ? 'Event Day'
                  : 'Baseline';

  const t0      = 20;
  const rawVc   = 0.3 + globalIntensity * 1.2;
  const vc      = staggeredArrivals ? Math.max(0.10, rawVc * 0.75) : rawVc;
  const tCurrent = t0 * (1 + 0.15 * Math.pow(vc, 4));
  const delayMin = tCurrent - t0;

  const congPct   = Math.round(metrics.congestionScore * 100);
  const congLabel = metrics.congestionScore > 0.7 ? 'Critical' : metrics.congestionScore > 0.4 ? 'Moderate' : 'Normal';

  const delayColor  = delayMin > 5 ? (delayMin > 12 ? '#FF453A' : '#FF9F0A') : '#30D158';
  const congColor   = metrics.congestionScore > 0.6 ? '#FF453A' : metrics.congestionScore > 0.3 ? '#FF9F0A' : '#30D158';
  const travelColor = tCurrent > 30 ? '#FF453A' : tCurrent > 24 ? '#FF9F0A' : 'rgba(255,255,255,0.85)';
  const co2Color    = metrics.co2Increase > 60 ? '#FF453A' : metrics.co2Increase > 30 ? '#FF9F0A' : 'rgba(255,255,255,0.85)';
  const personsVal  = metrics.estimatedPersonsAffected > 0
    ? `${Math.round(metrics.estimatedPersonsAffected / 1000)}K`
    : '—';

  const cells: { label: string; value: string; sub: string; color: string }[] = [
    { label: 'Avg. Delay',       value: `+${delayMin.toFixed(1)}`,  sub: 'min vs free-flow',       color: delayColor },
    { label: 'Congestion Score', value: `${congPct}`,               sub: congLabel,                 color: congColor  },
    { label: 'Travel Time',      value: `${tCurrent.toFixed(1)}`,   sub: `min · v/c ${vc.toFixed(2)}`, color: travelColor },
    { label: 'Peak Zones',       value: `${metrics.peakCongestionZones}`, sub: 'active surges',    color: 'rgba(255,255,255,0.85)' },
    { label: 'CO₂ Increase',    value: `+${metrics.co2Increase}%`,  sub: 'emissions delta',        color: co2Color   },
    { label: 'Persons Affected', value: personsVal,                  sub: 'peak hour est.',         color: metrics.estimatedPersonsAffected > 50000 ? '#FF9F0A' : 'rgba(255,255,255,0.85)' },
    { label: 'Transit Gap',      value: '80%',                       sub: 'underserved venues',     color: '#FF453A'  },
    { label: 'Parking Stress',   value: '~95%',                      sub: 'vs 44.5% baseline',     color: '#FF453A'  },
    { label: 'Collision Risk',   value: '5 PM',                      sub: '10,672 incidents',       color: '#FF453A'  },
    { label: 'Crime Index',      value: '65%',                       sub: 'property crimes',        color: '#FF9F0A'  },
  ];

  return (
    <div
      className="fixed right-4 z-40 flex flex-col gap-3 tilt-right"
      style={{ top: '60px', bottom: '124px', width: '268px', overflow: 'visible', marginTop: '4px' }}
    >
      {/* Mode badge */}
      <div
        className="flex items-center gap-2 px-3 py-2 rounded-xl flex-shrink-0 glass-surface"
      >
        <div className="rounded-full flex-shrink-0" style={{
          width: '6px', height: '6px',
          background: mode === 'crisis' ? '#FF453A' : mode === 'event' ? '#FF9F0A' : '#30D158',
        }} />
        <span style={{ fontSize: '12px', fontWeight: 500, color: 'rgba(255,255,255,0.6)', letterSpacing: '-0.01em' }}>
          {modeLabel}
        </span>
      </div>

      {/* Metrics grid */}
      <div
        className="flex-shrink-0 rounded-2xl overflow-hidden glass-panel tilt-right glass-shimmer-wrap"
      >
        <div style={{ padding: '10px 14px 6px', borderBottom: '0.5px solid rgba(255,255,255,0.06)' }}>
          <span style={{ fontSize: '11px', fontWeight: 600, color: 'rgba(255,255,255,0.3)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Live Metrics
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
          {cells.map(({ label, value, sub, color }, i) => {
            const lastRow = i >= cells.length - 2;
            const leftCol = i % 2 === 0;
            return (
              <div
                key={label}
                style={{
                  padding: '12px 14px',
                  borderBottom: lastRow ? 'none' : '0.5px solid rgba(255,255,255,0.04)',
                  borderRight: leftCol ? '0.5px solid rgba(255,255,255,0.04)' : 'none',
                }}
              >
                <div style={{
                  fontSize: '10px', color: 'rgba(255,255,255,0.4)', fontWeight: 500,
                  letterSpacing: '0.06em', textTransform: 'uppercase',
                  lineHeight: 1, marginBottom: '6px',
                }}>
                  {label}
                </div>
                <div style={{
                  fontSize: '24px', fontWeight: 200, color,
                  fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
                  lineHeight: 1, letterSpacing: '-0.02em',
                }}>
                  {value}
                </div>
                {sub && (
                  <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.25)', marginTop: '3px', lineHeight: 1 }}>
                    {sub}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Interventions */}
      {staggeredArrivals && (
        <div
          className="flex-shrink-0 rounded-2xl overflow-hidden glass-surface"
        >
          <div style={{ padding: '10px 14px 6px', borderBottom: '0.5px solid rgba(255,255,255,0.06)' }}>
            <span style={{ fontSize: '11px', fontWeight: 600, color: 'rgba(255,255,255,0.3)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              Interventions
            </span>
          </div>
          <div style={{ padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '13px', color: '#30D158', letterSpacing: '-0.01em' }}>✓ Staggered arrivals</span>
              <span style={{ fontSize: '12px', fontFamily: "'SF Mono', ui-monospace, monospace", color: '#30D158', fontWeight: 600 }}>
                −{(t0 * (1 + 0.15 * Math.pow(rawVc, 4)) - tCurrent).toFixed(1)} min
              </span>
            </div>
            {[
              { icon: '✓', color: '#30D158', label: 'Olympic Express Corridors',  status: 'Modeled'    },
              { icon: '⚠', color: '#FF9F0A', label: 'Long Beach transit gap',     status: 'Unresolved' },
              { icon: '●', color: '#FF453A', label: 'SoFi transit gap',           status: 'Unresolved' },
            ].map((r, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: r.color, flexShrink: 0, width: '12px' }}>{r.icon}</span>
                <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', flex: 1, letterSpacing: '-0.01em' }}>{r.label}</span>
                <span style={{ fontSize: '10px', fontWeight: 600, color: r.color, letterSpacing: '0.04em' }}>{r.status}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ML Predictions */}
      {mlPredictions && (
        <div className="flex-shrink-0 rounded-2xl overflow-hidden glass-surface">
          <div style={{ padding: '10px 14px 6px', borderBottom: '0.5px solid rgba(255,255,255,0.06)' }}>
            <span style={{ fontSize: '11px', fontWeight: 600, color: '#2DD4BF', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              ML Predictions
            </span>
          </div>
          <div style={{ padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {/* Row 1: Surge Forecast */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                  Surge Forecast
                </span>
                <span style={{ fontSize: '14px', fontWeight: 200, color: '#2DD4BF', letterSpacing: '-0.02em', fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif' }}>
                  {Math.round(mlPredictions.surgeIntensity * 100)}%
                </span>
              </div>
              {/* Confidence bar */}
              <div style={{ height: '3px', borderRadius: '2px', background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${Math.round(mlPredictions.confidence * 100)}%`,
                  background: '#2DD4BF',
                  borderRadius: '2px',
                  transition: 'width 0.5s ease',
                }} />
              </div>
              <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.2)', marginTop: '2px' }}>
                {Math.round(mlPredictions.confidence * 100)}% confidence
              </div>
            </div>

            {/* Row 2: Condition Class */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                Condition Class
              </span>
              <span style={{
                fontSize: '10px',
                fontWeight: 700,
                letterSpacing: '0.08em',
                padding: '2px 8px',
                borderRadius: '4px',
                background: `${CONDITION_COLORS[mlPredictions.condition]}20`,
                color: CONDITION_COLORS[mlPredictions.condition],
                border: `1px solid ${CONDITION_COLORS[mlPredictions.condition]}40`,
              }}>
                {mlPredictions.condition}
              </span>
            </div>

            {/* Model attribution */}
            <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.15)', lineHeight: 1.4 }}>
              Random Forest · Gradient Boosting · Trained on BPR simulations
            </div>
          </div>
        </div>
      )}

      {/* Congestion chart */}
      <div
        className="flex-1 min-h-0 rounded-2xl overflow-hidden flex flex-col glass-surface"
      >
        <div style={{ padding: '10px 14px 6px', borderBottom: '0.5px solid rgba(255,255,255,0.06)', flexShrink: 0 }}>
          <span style={{ fontSize: '11px', fontWeight: 600, color: 'rgba(255,255,255,0.3)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Congestion Over Time
          </span>
        </div>
        <div style={{ padding: '8px', flex: 1, minHeight: 0 }}>
          <CongestionChart />
        </div>
      </div>
    </div>
  );
}

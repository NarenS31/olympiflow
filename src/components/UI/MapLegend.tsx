import { useState } from 'react';

const LAYER_LEGEND = [
  { color: '#FF453A', label: 'Traffic Heatmap'    },
  { color: '#0A84FF', label: 'Olympic Venues'     },
  { color: '#BF5AF2', label: 'Transit Routes'     },
  { color: '#FF9F0A', label: 'Collision Hotspots' },
  { color: '#BF5AF2', label: 'Crime Density'      },
];

export function MapLegend() {
  const [open, setOpen] = useState(false);

  return (
    <div className="fixed z-40 flex flex-col items-end gap-2" style={{ bottom: '88px', right: '12px' }}>
      {open && (
        <div
          className="glass-panel"
          style={{
            width: '180px',
            borderRadius: '14px',
            overflow: 'hidden',
          }}
        >
          <div style={{ padding: '10px 14px 6px', borderBottom: '0.5px solid rgba(255,255,255,0.06)' }}>
            <span style={{ fontSize: '11px', fontWeight: 600, color: 'rgba(255,255,255,0.3)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              Layers
            </span>
          </div>

          {/* Risk gradient */}
          <div style={{ padding: '10px 14px', borderBottom: '0.5px solid rgba(255,255,255,0.06)' }}>
            <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.3)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '8px', fontWeight: 500 }}>
              Venue Risk
            </div>
            <div style={{ display: 'flex', gap: '2px', marginBottom: '5px' }}>
              {[
                { color: 'rgba(10,132,255,0.6)' },
                { color: 'rgba(255,159,10,0.7)' },
                { color: 'rgba(255,69,58,0.85)'  },
              ].map((r, i) => (
                <div
                  key={i}
                  style={{ flex: 1, height: '4px', borderRadius: '2px', background: r.color }}
                />
              ))}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              {['Low', 'Moderate', 'High'].map((l) => (
                <div key={l} style={{ fontSize: '9px', color: 'rgba(255,255,255,0.25)' }}>{l}</div>
              ))}
            </div>
          </div>

          {/* Layers */}
          <div style={{ padding: '10px 14px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {LAYER_LEGEND.map((l) => (
                <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: l.color, flexShrink: 0 }} />
                  <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', letterSpacing: '-0.01em' }}>{l.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Marker style note */}
          <div style={{ padding: '8px 14px 12px', borderTop: '0.5px solid rgba(255,255,255,0.06)' }}>
            <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.2)', lineHeight: 1.5 }}>
              Pin size = risk score<br />
              Red fill = highest-risk venue
            </div>
          </div>
        </div>
      )}

      {/* Toggle button */}
      <button
        onClick={() => setOpen((v) => !v)}
        title="Map legend"
        className="glass-surface"
        style={{
          width: '32px', height: '32px',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          borderRadius: '9px',
          color: open ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.4)',
          cursor: 'pointer',
          transition: 'all 0.2s',
        }}
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
          <rect x="3" y="3" width="7" height="7" rx="1"/>
          <rect x="14" y="3" width="7" height="7" rx="1"/>
          <rect x="3" y="14" width="7" height="7" rx="1"/>
          <line x1="14" y1="17.5" x2="21" y2="17.5"/>
          <line x1="17.5" y1="14" x2="17.5" y2="21"/>
        </svg>
      </button>
    </div>
  );
}

import { useState } from 'react';
import { useSimulationStore } from '../../stores/simulationStore';
import { LA28_VENUES } from '../../data/venues';
import { AIAdvisorSection } from './AIAdvisorSection';
import type { LayerVisibility, TrafficEventType } from '../../types';

// ── Venue risk dataset ────────────────────────────────────────────────────────
const VENUE_RISK: Record<string, { abbr: string; risk: number; athletes?: number; dashRoutes: number; riskColor: string }> = {
  'la-coliseum':      { abbr: 'COL',  risk: 78.1, athletes: 2269, dashRoutes: 1, riskColor: '#FF453A' },
  'long-beach-arena': { abbr: 'LB',   risk: 35.3, athletes: 546,  dashRoutes: 0, riskColor: '#FF9F0A' },
  'crypto-arena':     { abbr: 'CC',   risk: 35.0,                  dashRoutes: 5, riskColor: '#FF9F0A' },
  'sofi':             { abbr: 'SOFI', risk: 33.9,                  dashRoutes: 0, riskColor: '#FF9F0A' },
  'rose-bowl':        { abbr: 'RB',   risk: 33.9,                  dashRoutes: 0, riskColor: '#FF9F0A' },
};

const TOP_VENUES = [
  { id: 'la-coliseum',      name: 'LA Coliseum' },
  { id: 'long-beach-arena', name: 'Long Beach' },
  { id: 'crypto-arena',     name: 'Crypto.com' },
  { id: 'sofi',             name: 'SoFi' },
  { id: 'rose-bowl',        name: 'Rose Bowl' },
];

// ── BPR computation ───────────────────────────────────────────────────────────
function bprTravelTime(vc: number, t0 = 20) {
  return t0 * (1 + 0.15 * Math.pow(vc, 4));
}

function vcFromIntensity(intensity: number) {
  return 0.3 + intensity * 1.2;
}

function vcStatus(vc: number) {
  if (vc > 1.3) return { label: 'SEVERE CONGESTION', color: '#FF453A', tint: 'rgba(255,69,58,0.06)' };
  if (vc > 1.0) return { label: 'OVER CAPACITY',     color: '#FF453A', tint: 'rgba(255,69,58,0.06)' };
  if (vc > 0.9) return { label: 'NEAR CAPACITY',     color: '#FF9F0A', tint: 'rgba(255,159,10,0.06)' };
  if (vc > 0.7) return { label: 'MODERATE LOAD',     color: '#FF9F0A', tint: 'rgba(255,159,10,0.06)' };
  return           { label: 'FREE FLOW',             color: '#30D158', tint: 'rgba(48,209,88,0.06)' };
}

// ── Tab definitions ───────────────────────────────────────────────────────────
const TABS = [
  {
    id: 0,
    label: 'Layers',
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2"/>
        <polyline points="2 17 12 22 22 17"/>
        <polyline points="2 12 12 17 22 12"/>
      </svg>
    ),
  },
  {
    id: 1,
    label: 'BPR',
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    ),
  },
  {
    id: 2,
    label: 'Events',
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
      </svg>
    ),
  },
  {
    id: 3,
    label: 'AI',
    icon: (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>
        <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>
      </svg>
    ),
  },
];

// ── Sidebar ───────────────────────────────────────────────────────────────────
export function SimulationSidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState(0);

  if (collapsed) {
    return (
      <button
        onClick={() => setCollapsed(false)}
        className="fixed left-4 z-40 flex items-center justify-center glass-panel"
        style={{
          top: '56px',
          marginTop: '12px',
          width: '36px', height: '36px',
          borderRadius: '12px',
          color: 'rgba(255,255,255,0.5)',
          cursor: 'pointer',
        }}
        title="Open controls"
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
          <line x1="3" y1="12" x2="21" y2="12"/>
          <line x1="3" y1="6"  x2="21" y2="6"/>
          <line x1="3" y1="18" x2="21" y2="18"/>
        </svg>
      </button>
    );
  }

  return (
    <aside
      className="fixed left-4 z-40 flex flex-col glass-panel tilt-left glass-shimmer-wrap"
      style={{
        top: '56px',
        bottom: '112px',
        width: '260px',
        borderRadius: '16px',
        overflow: 'hidden',
        marginTop: '8px',
      }}
    >
      {/* Panel header */}
      <div
        className="flex items-center justify-between px-4 py-2.5 flex-shrink-0"
        style={{ borderBottom: '0.5px solid rgba(255,255,255,0.08)' }}
      >
        <div className="flex items-center gap-2">
          <div
            className="rounded-full flex-shrink-0 animate-status-blink"
            style={{ width: '6px', height: '6px', background: '#30D158', boxShadow: '0 0 6px #30D158' }}
          />
          <span style={{
            fontSize: '11px', fontWeight: 600,
            color: 'rgba(255,255,255,0.3)',
            letterSpacing: '0.08em', textTransform: 'uppercase',
          }}>
            LA28 Transport
          </span>
        </div>
        <button
          onClick={() => setCollapsed(true)}
          style={{ color: 'rgba(255,255,255,0.25)', background: 'none', border: 'none', cursor: 'pointer', display: 'flex' }}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="15 18 9 12 15 6"/>
          </svg>
        </button>
      </div>

      {/* Tab row */}
      <div
        className="flex items-center px-3 flex-shrink-0"
        style={{ borderBottom: '0.5px solid rgba(255,255,255,0.06)', height: '44px', gap: '4px' }}
      >
        {TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              title={tab.label}
              style={{
                flex: 1,
                height: '36px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: '8px',
                border: 'none',
                background: 'transparent',
                color: isActive ? '#0A84FF' : 'rgba(255,255,255,0.3)',
                cursor: 'pointer',
                transition: 'color 0.15s',
              }}
              onMouseEnter={(e) => {
                if (!isActive) (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)';
              }}
              onMouseLeave={(e) => {
                if (!isActive) (e.currentTarget as HTMLElement).style.background = 'transparent';
              }}
            >
              {tab.icon}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <div
        className="flex-1 scrollbar-none px-4 py-4 space-y-5"
        style={{ overflowY: 'auto' }}
      >
        {activeTab === 0 && (
          <>
            <LayerSection />
            <VenueRiskSection />
          </>
        )}
        {activeTab === 1 && (
          <>
            <BPRMonitorSection />
            <ArrivalStaggeringSection />
            <IntensitySection />
          </>
        )}
        {activeTab === 2 && (
          <>
            <CustomEventsSection />
            <VenueSurgeSection />
            <ActionsSection />
          </>
        )}
        {activeTab === 3 && (
          <AIAdvisorSection />
        )}
      </div>
    </aside>
  );
}

// ── Layer section ─────────────────────────────────────────────────────────────
const LAYER_DEFS: { key: keyof LayerVisibility; label: string; color: string }[] = [
  { key: 'heatmap',    label: 'Traffic Heatmap',    color: '#FF453A' },
  { key: 'venues',     label: 'Olympic Venues',     color: '#0A84FF' },
  { key: 'transit',    label: 'Transit Routes',     color: '#BF5AF2' },
  { key: 'collisions', label: 'Collision Hotspots', color: '#FF9F0A' },
  { key: 'crime',      label: 'Crime Density',      color: '#BF5AF2' },
  { key: 'parking',    label: 'Parking Stress',     color: '#FF9F0A' },
];

function LayerSection() {
  const layers      = useSimulationStore((s) => s.layers);
  const toggleLayer = useSimulationStore((s) => s.toggleLayer);

  return (
    <AppleSection title="Layers">
      <div>
        {LAYER_DEFS.map((def, i) => {
          const active = layers[def.key];
          const isLast = i === LAYER_DEFS.length - 1;
          return (
            <button
              key={def.key}
              onClick={() => toggleLayer(def.key)}
              className="w-full flex items-center gap-3 transition-colors"
              style={{
                height: '40px',
                background: 'transparent',
                border: 'none',
                borderBottom: isLast ? 'none' : '0.5px solid rgba(255,255,255,0.06)',
                cursor: 'pointer',
                padding: '0 2px',
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)'; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
            >
              <div
                className="rounded-full flex-shrink-0"
                style={{ width: '8px', height: '8px', background: def.color, opacity: active ? 1 : 0.25 }}
              />
              <span style={{
                fontSize: '13px',
                color: active ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.35)',
                fontWeight: 400,
                flex: 1,
                textAlign: 'left',
                transition: 'color 0.15s',
                letterSpacing: '-0.01em',
              }}>
                {def.label}
              </span>
              {/* Apple toggle */}
              <div
                className={`apple-toggle ${active ? 'active' : ''}`}
                style={{
                  background: active ? '#30D158' : 'rgba(255,255,255,0.12)',
                  width: '36px', height: '22px',
                  borderRadius: '11px',
                }}
              >
                <div
                  className="apple-toggle-thumb"
                  style={{
                    width: '18px', height: '18px',
                    top: '2px', left: '2px',
                    transform: active ? 'translateX(14px)' : 'none',
                  }}
                />
              </div>
            </button>
          );
        })}
      </div>
    </AppleSection>
  );
}

// ── Venue Risk Index table ────────────────────────────────────────────────────
function VenueRiskSection() {
  return (
    <AppleSection title="Venue Risk Index">
      <div>
        <div
          className="grid"
          style={{
            gridTemplateColumns: '2fr 1.2fr 1.4fr 1fr',
            padding: '4px 0 8px',
            borderBottom: '0.5px solid rgba(255,255,255,0.06)',
          }}
        >
          {['VENUE', 'RISK', 'ATH', 'DASH'].map((h) => (
            <span key={h} style={{
              fontSize: '10px', color: 'rgba(255,255,255,0.3)',
              fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase',
            }}>
              {h}
            </span>
          ))}
        </div>
        {TOP_VENUES.map(({ id }, i) => {
          const d = VENUE_RISK[id];
          if (!d) return null;
          const isColiseum = id === 'la-coliseum';
          const isLast = i === TOP_VENUES.length - 1;
          return (
            <div
              key={id}
              className="grid items-center"
              style={{
                gridTemplateColumns: '2fr 1.2fr 1.4fr 1fr',
                height: '36px',
                borderBottom: isLast ? 'none' : '0.5px solid rgba(255,255,255,0.04)',
                background: isColiseum ? 'rgba(255,69,58,0.06)' : i % 2 !== 0 ? 'transparent' : 'rgba(255,255,255,0.01)',
              }}
            >
              <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.7)', fontWeight: 500 }}>{d.abbr}</span>
              <span style={{
                fontSize: '12px', color: d.riskColor, fontWeight: 600,
                fontFamily: "'SF Mono', ui-monospace, monospace",
              }}>
                {d.risk.toFixed(1)}
              </span>
              <span style={{
                fontSize: '12px', color: 'rgba(255,255,255,0.4)',
                fontFamily: "'SF Mono', ui-monospace, monospace",
              }}>
                {d.athletes ? d.athletes.toLocaleString() : <span style={{ color: 'rgba(255,255,255,0.2)' }}>—</span>}
              </span>
              <span style={{
                fontSize: '12px',
                fontFamily: "'SF Mono', ui-monospace, monospace",
                color: d.dashRoutes > 0 ? '#0A84FF' : 'rgba(255,255,255,0.2)',
              }}>
                {d.dashRoutes > 0 ? `${d.dashRoutes}` : '—'}
              </span>
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.2)', marginTop: '6px', letterSpacing: '0.02em' }}>
        DASH = LADOT routes serving venue
      </div>
    </AppleSection>
  );
}

// ── BPR Monitor ───────────────────────────────────────────────────────────────
function BPRMonitorSection() {
  const globalIntensity   = useSimulationStore((s) => s.globalIntensity);
  const staggeredArrivals = useSimulationStore((s) => s.staggeredArrivals);

  const vc        = vcFromIntensity(globalIntensity);
  const baseTime  = bprTravelTime(vc);
  const stagVC    = vc * 0.75;
  const stagTime  = bprTravelTime(stagVC);
  const savings   = baseTime - stagTime;
  const savingsPct = (savings / baseTime * 100);

  const displayVC = staggeredArrivals ? stagVC   : vc;
  const displayT  = staggeredArrivals ? stagTime : baseTime;
  const status    = vcStatus(displayVC);

  const bprResult  = (1 + 0.15 * Math.pow(displayVC, 4));
  const formulaStr = `t = [1 + 0.15 × (${displayVC.toFixed(2)})⁴] = ${bprResult.toFixed(2)}×`;

  return (
    <AppleSection title="BPR Monitor">
      {/* Big numbers */}
      <div className="flex gap-4 mb-4">
        <div>
          <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '4px' }}>
            Corridor V/C
          </div>
          <div style={{
            fontSize: '32px', fontWeight: 200, color: status.color,
            fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
            lineHeight: 1, letterSpacing: '-0.02em',
          }}>
            {displayVC.toFixed(2)}
          </div>
        </div>
        <div>
          <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '4px' }}>
            Travel Time
          </div>
          <div style={{
            fontSize: '32px', fontWeight: 200, color: 'rgba(255,255,255,0.85)',
            fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
            lineHeight: 1, letterSpacing: '-0.02em',
          }}>
            {displayT.toFixed(1)}
            <span style={{ fontSize: '14px', fontWeight: 400, color: 'rgba(255,255,255,0.4)', marginLeft: '3px' }}>min</span>
          </div>
        </div>
      </div>

      {/* Status pill */}
      <div className="flex items-center justify-between mb-3">
        <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.3)', fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Status</span>
        <div
          style={{
            padding: '3px 10px',
            borderRadius: '20px',
            background: status.tint,
            fontSize: '10px',
            fontWeight: 600,
            color: status.color,
            letterSpacing: '0.04em',
          }}
        >
          {status.label}
        </div>
      </div>

      {/* Formula */}
      <div style={{
        fontSize: '10px',
        fontFamily: "'SF Mono', ui-monospace, monospace",
        color: 'rgba(255,255,255,0.25)',
        letterSpacing: '0.02em',
        padding: '8px 0',
        borderTop: '0.5px solid rgba(255,255,255,0.06)',
      }}>
        {formulaStr}
      </div>

      {/* Savings row */}
      {staggeredArrivals && (
        <div className="flex items-center justify-between mt-2" style={{ borderTop: '0.5px solid rgba(255,255,255,0.06)', paddingTop: '8px' }}>
          <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Savings</span>
          <span style={{
            fontSize: '12px', fontWeight: 600, color: '#30D158',
            fontFamily: "'SF Mono', ui-monospace, monospace",
          }}>
            −{savings.toFixed(1)} min (−{savingsPct.toFixed(1)}%)
          </span>
        </div>
      )}
    </AppleSection>
  );
}

// ── Arrival Staggering toggle ─────────────────────────────────────────────────
function ArrivalStaggeringSection() {
  const staggeredArrivals       = useSimulationStore((s) => s.staggeredArrivals);
  const toggleStaggeredArrivals = useSimulationStore((s) => s.toggleStaggeredArrivals);

  return (
    <AppleSection title="Arrival Staggering">
      <div className="flex items-center justify-between" style={{ height: '44px' }}>
        <div>
          <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.85)', fontWeight: 400, letterSpacing: '-0.01em' }}>
            Staggered Arrivals
          </div>
          <div style={{
            fontSize: '11px',
            color: staggeredArrivals ? '#30D158' : 'rgba(255,255,255,0.3)',
            marginTop: '2px',
            fontWeight: 400,
          }}>
            {staggeredArrivals ? 'Active' : 'Off'}
          </div>
        </div>
        <button
          onClick={toggleStaggeredArrivals}
          className={`apple-toggle ${staggeredArrivals ? 'active' : ''}`}
          style={{ background: staggeredArrivals ? '#30D158' : 'rgba(120,120,128,0.32)' }}
        >
          <div className="apple-toggle-thumb" />
        </button>
      </div>

      {staggeredArrivals && (
        <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '0.5px solid rgba(255,255,255,0.06)' }}>
          {[
            { label: 'Baseline',   vc: '1.29', t: '28.3 min', color: '#FF453A' },
            { label: 'Staggered',  vc: '0.97', t: '22.6 min', color: '#30D158', delta: '−5.7 min' },
          ].map((row) => (
            <div
              key={row.label}
              className="flex items-center gap-2 mb-1.5"
              style={{ padding: '6px 0' }}
            >
              <div className="rounded-full flex-shrink-0" style={{ width: '6px', height: '6px', background: row.color }} />
              <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)', width: '58px', fontWeight: 500 }}>
                {row.label}
              </span>
              <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', fontFamily: "'SF Mono', ui-monospace, monospace" }}>
                v/c {row.vc}
              </span>
              <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', fontFamily: "'SF Mono', ui-monospace, monospace", flex: 1 }}>
                {row.t}
              </span>
              {row.delta && (
                <span style={{ fontSize: '11px', fontWeight: 600, color: '#30D158', fontFamily: "'SF Mono', ui-monospace, monospace" }}>
                  {row.delta}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </AppleSection>
  );
}

// ── Traffic Intensity slider ──────────────────────────────────────────────────
function IntensitySection() {
  const globalIntensity    = useSimulationStore((s) => s.globalIntensity);
  const setGlobalIntensity = useSimulationStore((s) => s.setGlobalIntensity);

  const pct   = Math.round(globalIntensity * 100);
  const color = globalIntensity > 0.7 ? '#FF453A' : globalIntensity > 0.4 ? '#FF9F0A' : '#0A84FF';
  const label = globalIntensity > 0.7 ? 'Surge' : globalIntensity > 0.4 ? 'Elevated' : 'Nominal';

  return (
    <AppleSection title="Traffic Intensity">
      <div>
        <div className="flex items-baseline justify-between mb-3">
          <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', fontWeight: 400, letterSpacing: '-0.01em' }}>
            Network Pressure
          </span>
          <div className="flex items-baseline gap-1.5">
            <span style={{ fontSize: '11px', color, fontWeight: 500 }}>{label}</span>
            <span style={{
              fontSize: '20px', fontWeight: 200, color,
              fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
              letterSpacing: '-0.02em',
            }}>
              {pct}%
            </span>
          </div>
        </div>
        <input
          type="range"
          min={0} max={1} step={0.01}
          value={globalIntensity}
          onChange={(e) => setGlobalIntensity(parseFloat(e.target.value))}
          className="w-full appearance-none cursor-pointer"
          style={{
            height: '4px',
            borderRadius: '2px',
            background: `linear-gradient(to right, ${color} ${pct}%, rgba(255,255,255,0.08) ${pct}%)`,
          }}
        />
        <div className="flex justify-between mt-2">
          <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.2)' }}>Low</span>
          <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.2)' }}>Peak</span>
        </div>
      </div>
    </AppleSection>
  );
}

// ── Custom Traffic Events ─────────────────────────────────────────────────────
const EVT_COLORS: Record<TrafficEventType, string> = {
  sports:  '#0A84FF',
  concert: '#BF5AF2',
  festival:'#30D158',
  rally:   '#FF9F0A',
};

const EVT_DECAY: Record<TrafficEventType, string> = {
  sports:  '22 km radius',
  concert: '22 km radius',
  festival:'28 km radius',
  rally:   '32 km radius',
};

function fmtAtt(n: number) {
  return n >= 1000 ? `${Math.round(n / 1000)}k` : String(n);
}

function CustomEventsSection() {
  const [showForm, setShowForm]       = useState(false);
  const [pendingName, setPendingName] = useState('');
  const [pendingAtt, setPendingAtt]   = useState(25000);
  const [pendingType, setPendingType] = useState<TrafficEventType>('sports');

  const customEvents            = useSimulationStore((s) => s.customEvents);
  const placingEvent            = useSimulationStore((s) => s.placingEvent);
  const pendingEventLocation    = useSimulationStore((s) => s.pendingEventLocation);
  const addCustomEvent          = useSimulationStore((s) => s.addCustomEvent);
  const removeCustomEvent       = useSimulationStore((s) => s.removeCustomEvent);
  const setPlacingEvent         = useSimulationStore((s) => s.setPlacingEvent);
  const setPendingEventLocation = useSimulationStore((s) => s.setPendingEventLocation);

  const reset = () => {
    setShowForm(false);
    setPlacingEvent(false);
    setPendingName('');
    setPendingAtt(25000);
    setPendingType('sports');
  };

  const confirm = () => {
    if (!pendingEventLocation) return;
    addCustomEvent({
      id: `evt-${Date.now()}`,
      name: pendingName.trim() || `${pendingType.charAt(0).toUpperCase() + pendingType.slice(1)} Event`,
      lng: pendingEventLocation.lng,
      lat: pendingEventLocation.lat,
      attendees: pendingAtt,
      type: pendingType,
    });
    reset();
  };

  const sliderPct = ((pendingAtt - 5000) / 95000) * 100;

  return (
    <AppleSection title="Traffic Events">
      <div className="space-y-2">
        {customEvents.map((evt) => (
          <div
            key={evt.id}
            className="flex items-start gap-2.5 py-2"
            style={{ borderBottom: '0.5px solid rgba(255,255,255,0.06)' }}
          >
            <div className="rounded-full flex-shrink-0 mt-1" style={{ width: '6px', height: '6px', background: EVT_COLORS[evt.type] }} />
            <div className="flex-1 min-w-0">
              <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.85)', fontWeight: 400, letterSpacing: '-0.01em' }} className="truncate">
                {evt.name}
              </div>
              <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', marginTop: '1px' }}>
                {fmtAtt(evt.attendees)} · {evt.type} · {EVT_DECAY[evt.type]}
              </div>
            </div>
            <button
              onClick={() => removeCustomEvent(evt.id)}
              style={{ color: 'rgba(255,255,255,0.2)', fontSize: '14px', background: 'none', border: 'none', cursor: 'pointer', lineHeight: 1 }}
            >
              ×
            </button>
          </div>
        ))}

        {showForm ? (
          <div className="space-y-3 pt-1">
            <input
              type="text"
              placeholder="Event name (optional)"
              value={pendingName}
              onChange={(e) => setPendingName(e.target.value)}
              className="w-full outline-none"
              style={{
                background: 'rgba(255,255,255,0.06)',
                border: '0.5px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                padding: '8px 12px',
                fontSize: '13px',
                color: 'rgba(255,255,255,0.85)',
                fontFamily: 'inherit',
              }}
            />

            <div>
              <div className="flex justify-between mb-2">
                <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Attendees</span>
                <span style={{ fontSize: '13px', fontWeight: 600, color: 'rgba(255,255,255,0.85)', fontFamily: "'SF Mono', ui-monospace, monospace" }}>{fmtAtt(pendingAtt)}</span>
              </div>
              <input
                type="range" min={5000} max={100000} step={1000}
                value={pendingAtt}
                onChange={(e) => setPendingAtt(parseInt(e.target.value))}
                className="w-full appearance-none cursor-pointer"
                style={{
                  height: '4px', borderRadius: '2px',
                  background: `linear-gradient(to right, ${EVT_COLORS[pendingType]} ${sliderPct}%, rgba(255,255,255,0.08) ${sliderPct}%)`,
                }}
              />
            </div>

            <div>
              <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)', fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '8px' }}>Event Type</div>
              <div className="grid grid-cols-2 gap-1.5">
                {(['sports', 'concert', 'festival', 'rally'] as TrafficEventType[]).map((type) => (
                  <button
                    key={type}
                    onClick={() => setPendingType(type)}
                    className="py-2 rounded-lg transition-all capitalize"
                    style={
                      pendingType === type
                        ? { background: `${EVT_COLORS[type]}18`, border: `0.5px solid ${EVT_COLORS[type]}80`, color: EVT_COLORS[type], fontSize: '12px', fontWeight: 500, fontFamily: 'inherit' }
                        : { background: 'rgba(255,255,255,0.04)', border: '0.5px solid rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.4)', fontSize: '12px', fontFamily: 'inherit' }
                    }
                  >{type}</button>
                ))}
              </div>
            </div>

            {!pendingEventLocation ? (
              <button
                onClick={() => setPlacingEvent(true)}
                className="w-full py-2.5 rounded-lg transition-all"
                style={placingEvent
                  ? { border: '0.5px solid rgba(255,159,10,0.4)', color: '#FF9F0A', background: 'rgba(255,159,10,0.08)', fontSize: '13px', fontFamily: 'inherit' }
                  : { border: '0.5px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.5)', fontSize: '13px', fontFamily: 'inherit', background: 'transparent' }
                }
              >
                {placingEvent ? 'Click map to place…' : 'Place on Map'}
              </button>
            ) : (
              <div className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ background: 'rgba(48,209,88,0.08)', border: '0.5px solid rgba(48,209,88,0.2)' }}>
                <span style={{ fontSize: '11px', color: '#30D158', fontWeight: 600 }}>Placed</span>
                <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', fontFamily: "'SF Mono', ui-monospace, monospace", flex: 1 }}>
                  {pendingEventLocation.lat.toFixed(3)}, {pendingEventLocation.lng.toFixed(3)}
                </span>
                <button
                  onClick={() => { setPendingEventLocation(null); setPlacingEvent(true); }}
                  style={{ fontSize: '11px', color: '#0A84FF', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'inherit' }}
                >
                  Re-place
                </button>
              </div>
            )}

            <div className="flex gap-2">
              <button
                onClick={confirm}
                disabled={!pendingEventLocation}
                className="flex-1 py-2.5 rounded-lg transition-all disabled:opacity-30"
                style={{ background: 'rgba(10,132,255,0.12)', border: '0.5px solid rgba(10,132,255,0.3)', color: '#0A84FF', fontSize: '13px', fontFamily: 'inherit', fontWeight: 500 }}
              >
                Simulate Impact
              </button>
              <button
                onClick={reset}
                className="px-4 py-2.5 rounded-lg"
                style={{ border: '0.5px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.4)', fontSize: '13px', fontFamily: 'inherit', background: 'transparent' }}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => setShowForm(true)}
            className="w-full py-2.5 rounded-lg transition-all"
            style={{
              border: '0.5px dashed rgba(255,255,255,0.12)',
              color: 'rgba(255,255,255,0.3)',
              fontSize: '13px',
              fontFamily: 'inherit',
              background: 'transparent',
              cursor: 'pointer',
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = '#0A84FF'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.3)'; }}
          >
            + Add Traffic Event
          </button>
        )}
      </div>
    </AppleSection>
  );
}

// ── Venue Surge ───────────────────────────────────────────────────────────────
function VenueSurgeSection() {
  const venueSurges    = useSimulationStore((s) => s.venueSurges);
  const setVenueSurge  = useSimulationStore((s) => s.setVenueSurge);
  const removeVenueSurge = useSimulationStore((s) => s.removeVenueSurge);

  return (
    <AppleSection title="Venue Crowd Surge">
      <div className="space-y-2">
        {LA28_VENUES.slice(0, 6).map((venue) => {
          const surge    = venueSurges[venue.id] ?? 0;
          const isActive = surge > 0;
          const surgeColor = surge > 0.6 ? '#FF453A' : surge > 0.3 ? '#FF9F0A' : '#0A84FF';
          return (
            <div key={venue.id}>
              <div className="flex items-center gap-2" style={{ height: '32px' }}>
                <span style={{
                  fontSize: '13px',
                  color: isActive ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.35)',
                  fontWeight: 400,
                  flex: 1,
                  letterSpacing: '-0.01em',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}>
                  {venue.shortName}
                </span>
                {isActive && (
                  <>
                    <span style={{ fontSize: '11px', color: surgeColor, fontFamily: "'SF Mono', ui-monospace, monospace", fontWeight: 600 }}>
                      {Math.round(surge * 100)}%
                    </span>
                    <button
                      onClick={() => removeVenueSurge(venue.id)}
                      style={{ color: 'rgba(255,255,255,0.2)', fontSize: '14px', background: 'none', border: 'none', cursor: 'pointer', lineHeight: 1 }}
                    >
                      ×
                    </button>
                  </>
                )}
              </div>
              {isActive ? (
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={surge}
                  onChange={(e) => setVenueSurge(venue.id, parseFloat(e.target.value))}
                  className="w-full appearance-none cursor-pointer"
                  style={{
                    height: '4px', borderRadius: '2px',
                    background: `linear-gradient(to right, ${surgeColor} ${surge * 100}%, rgba(255,255,255,0.08) ${surge * 100}%)`,
                  }}
                />
              ) : (
                <button
                  onClick={() => setVenueSurge(venue.id, 0.5)}
                  style={{
                    fontSize: '11px',
                    color: 'rgba(255,255,255,0.2)',
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontFamily: 'inherit',
                    padding: 0,
                  }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = '#0A84FF'; }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.2)'; }}
                >
                  + Add surge
                </button>
              )}
            </div>
          );
        })}
      </div>
    </AppleSection>
  );
}

// ── Quick Actions ─────────────────────────────────────────────────────────────
function ActionsSection() {
  const resetSimulation    = useSimulationStore((s) => s.resetSimulation);
  const setGlobalIntensity = useSimulationStore((s) => s.setGlobalIntensity);
  const setVenueSurge      = useSimulationStore((s) => s.setVenueSurge);

  const actions = [
    {
      label: 'Inject Traffic Surge',
      color: '#FF9F0A',
      icon: '⚡',
      onClick: () => setGlobalIntensity(0.9),
    },
    {
      label: 'Mass Venue Event',
      color: '#FF453A',
      icon: '◈',
      onClick: () => {
        setVenueSurge('sofi', 0.85);
        setVenueSurge('crypto-arena', 0.7);
        setVenueSurge('rose-bowl', 0.8);
        setGlobalIntensity(0.85);
      },
    },
    {
      label: 'Reset Simulation',
      color: 'rgba(255,255,255,0.3)',
      icon: '↺',
      onClick: resetSimulation,
    },
  ];

  return (
    <AppleSection title="Quick Actions">
      <div>
        {actions.map((a, i) => (
          <button
            key={i}
            onClick={a.onClick}
            className="w-full flex items-center gap-3 transition-colors"
            style={{
              height: '40px',
              background: 'transparent',
              border: 'none',
              borderBottom: i < actions.length - 1 ? '0.5px solid rgba(255,255,255,0.06)' : 'none',
              cursor: 'pointer',
              padding: '0 2px',
              color: a.color,
              textAlign: 'left',
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
          >
            <span style={{ fontSize: '14px', lineHeight: 1, width: '16px', textAlign: 'center' }}>{a.icon}</span>
            <span style={{ fontSize: '13px', fontWeight: 400, letterSpacing: '-0.01em' }}>{a.label}</span>
          </button>
        ))}
      </div>
    </AppleSection>
  );
}

// ── Shared section wrapper ────────────────────────────────────────────────────
function AppleSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ marginBottom: '12px' }}>
        <span style={{
          fontSize: '11px', fontWeight: 600,
          color: 'rgba(255,255,255,0.3)',
          letterSpacing: '0.08em', textTransform: 'uppercase',
        }}>
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}

import { useState } from 'react';
import { useSimulationStore } from '../../stores/simulationStore';
import { LA28_VENUES } from '../../data/venues';
import type { LayerVisibility } from '../../types';

export function SimulationSidebar() {
  const [collapsed, setCollapsed] = useState(false);

  if (collapsed) {
    return (
      <button
        onClick={() => setCollapsed(false)}
        className="fixed left-4 top-20 z-40 w-9 h-9 bg-surface-800/90 border border-surface-700/60 rounded-lg flex items-center justify-center text-surface-500 hover:text-slate-300 backdrop-blur-xl transition-colors"
        title="Open controls"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/>
        </svg>
      </button>
    );
  }

  return (
    <aside className="fixed left-4 top-[68px] bottom-[100px] z-40 w-72 flex flex-col bg-surface-900/90 backdrop-blur-xl border border-surface-700/50 rounded-xl overflow-hidden shadow-2xl">
      {/* Sidebar header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-700/50 flex-shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 bg-slate-500 rounded-full" />
          <span className="text-[12px] font-semibold text-slate-300 uppercase tracking-widest">
            Controls
          </span>
        </div>
        <button
          onClick={() => setCollapsed(true)}
          className="text-surface-600 hover:text-slate-300 transition-colors"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="15 18 9 12 15 6"/>
          </svg>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto overscroll-contain scrollbar-thin px-4 py-3 space-y-5">
        <LayerSection />
        <IntensitySection />
        <VenueSection />
        <ActionsSection />
      </div>
    </aside>
  );
}

const LAYER_COLORS: Record<string, string> = {
  heatmap: '#9a3412',
  venues: '#1d4ed8',
  transit: '#4c1d95',
  parking: '#78350f',
  collisions: '#7f1d1d',
};

function LayerSection() {
  const layers = useSimulationStore((s) => s.layers);
  const toggleLayer = useSimulationStore((s) => s.toggleLayer);

  const items: { key: keyof LayerVisibility; label: string }[] = [
    { key: 'heatmap', label: 'Traffic Heatmap' },
    { key: 'venues', label: 'Olympic Venues' },
    { key: 'transit', label: 'Transit Routes' },
    { key: 'parking', label: 'Parking Density' },
    { key: 'collisions', label: 'Collision Hotspots' },
  ];

  return (
    <Section title="Layers">
      <div className="space-y-1.5">
        {items.map((item) => {
          const active = layers[item.key];
          const color = LAYER_COLORS[item.key];
          return (
            <button
              key={item.key}
              onClick={() => toggleLayer(item.key)}
              className="w-full flex items-center gap-2.5 p-2.5 rounded-lg bg-surface-800/40 hover:bg-surface-700/50 transition-colors border border-surface-700/30"
            >
              <div
                className="w-3 h-3 rounded-sm flex-shrink-0 transition-all"
                style={{
                  backgroundColor: active ? color : 'transparent',
                  border: `1.5px solid ${active ? color : '#334155'}`,
                  opacity: active ? 0.85 : 1,
                }}
              />
              <span className={`text-[12px] flex-1 text-left ${active ? 'text-slate-200' : 'text-surface-600'}`}>
                {item.label}
              </span>
            </button>
          );
        })}
      </div>
    </Section>
  );
}

function IntensitySection() {
  const globalIntensity = useSimulationStore((s) => s.globalIntensity);
  const setGlobalIntensity = useSimulationStore((s) => s.setGlobalIntensity);

  const getColor = (v: number) =>
    v > 0.7 ? '#b91c1c' : v > 0.4 ? '#b45309' : '#3d5068';

  return (
    <Section title="Global Intensity">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] text-surface-500">Traffic Pressure</span>
          <span className="text-[13px] font-mono font-bold" style={{ color: getColor(globalIntensity) }}>
            {Math.round(globalIntensity * 100)}%
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={globalIntensity}
          onChange={(e) => setGlobalIntensity(parseFloat(e.target.value))}
          className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
          style={{
            background: `linear-gradient(to right, ${getColor(globalIntensity)} ${globalIntensity * 100}%, #1e293b ${globalIntensity * 100}%)`,
          }}
        />
        <div className="flex justify-between text-[10px] text-surface-600">
          <span>Low</span>
          <span>Peak</span>
        </div>
      </div>
    </Section>
  );
}

function VenueSection() {
  const venueSurges = useSimulationStore((s) => s.venueSurges);
  const setVenueSurge = useSimulationStore((s) => s.setVenueSurge);
  const removeVenueSurge = useSimulationStore((s) => s.removeVenueSurge);

  return (
    <Section title="Venue Crowd Surge">
      <div className="space-y-1.5">
        {LA28_VENUES.slice(0, 7).map((venue) => {
          const surge = venueSurges[venue.id] ?? 0;
          const isActive = surge > 0;
          return (
            <div
              key={venue.id}
              className="rounded-lg border border-surface-700/30 overflow-hidden"
            >
              <div className="flex items-center gap-2 px-2.5 py-2 bg-surface-800/40">
                <div
                  className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: isActive ? '#9a3412' : '#1e293b' }}
                />
                <span className="text-[11px] text-slate-300 flex-1 truncate font-medium">
                  {venue.shortName}
                </span>
                {isActive && (
                  <button
                    onClick={() => removeVenueSurge(venue.id)}
                    className="text-surface-600 hover:text-slate-400 text-[10px] transition-colors"
                  >
                    ✕
                  </button>
                )}
              </div>
              {isActive && (
                <div className="px-2.5 pb-2 bg-surface-800/20">
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={surge}
                    onChange={(e) => setVenueSurge(venue.id, parseFloat(e.target.value))}
                    className="w-full h-1 rounded-full appearance-none cursor-pointer mt-1.5"
                    style={{
                      background: `linear-gradient(to right, #9a3412 ${surge * 100}%, #1e293b ${surge * 100}%)`,
                    }}
                  />
                  <div className="flex justify-between text-[10px] mt-1">
                    <span className="text-surface-600">Crowd surge</span>
                    <span className="text-red-500 font-mono">{Math.round(surge * 100)}%</span>
                  </div>
                </div>
              )}
              {!isActive && (
                <button
                  onClick={() => setVenueSurge(venue.id, 0.5)}
                  className="w-full text-[10px] text-surface-600 hover:text-slate-400 py-1.5 transition-colors bg-surface-800/20"
                >
                  + Add surge
                </button>
              )}
            </div>
          );
        })}
      </div>
    </Section>
  );
}

function ActionsSection() {
  const resetSimulation = useSimulationStore((s) => s.resetSimulation);
  const setGlobalIntensity = useSimulationStore((s) => s.setGlobalIntensity);
  const setVenueSurge = useSimulationStore((s) => s.setVenueSurge);

  return (
    <Section title="Actions">
      <div className="space-y-2">
        <ActionBtn
          label="Inject Traffic Surge"
          color="amber"
          onClick={() => setGlobalIntensity(0.9)}
        />
        <ActionBtn
          label="Mass Venue Event"
          color="red"
          onClick={() => {
            setVenueSurge('sofi', 0.85);
            setVenueSurge('crypto-arena', 0.7);
            setVenueSurge('rose-bowl', 0.8);
            setGlobalIntensity(0.85);
          }}
        />
        <ActionBtn
          label="Reset Simulation"
          color="slate"
          onClick={resetSimulation}
        />
      </div>
    </Section>
  );
}

function ActionBtn({
  label, color, onClick,
}: {
  label: string; color: string; onClick: () => void;
}) {
  const colorMap: Record<string, string> = {
    amber: 'border-amber-900/60 text-amber-600 hover:bg-amber-950/40 hover:text-amber-500',
    red: 'border-red-900/60 text-red-700 hover:bg-red-950/40 hover:text-red-500',
    slate: 'border-surface-700/50 text-surface-500 hover:bg-surface-700/30 hover:text-slate-300',
  };
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg border text-[12px] font-medium transition-all ${colorMap[color]}`}
    >
      <span>{label}</span>
    </button>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[10px] font-semibold text-surface-600 uppercase tracking-widest mb-2">
        {title}
      </div>
      {children}
    </div>
  );
}

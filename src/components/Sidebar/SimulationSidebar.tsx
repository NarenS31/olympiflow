import { useState } from 'react';
import { useSimulationStore } from '../../stores/simulationStore';
import { LA28_VENUES } from '../../data/venues';
import type { LayerVisibility, TrafficEventType } from '../../types';

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
        <CustomEventsSection />
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

const EVENT_TYPE_COLORS: Record<TrafficEventType, string> = {
  sports:  '#0891b2',
  concert: '#7c3aed',
  festival:'#16a34a',
  rally:   '#d97706',
};

const EVENT_TYPE_DECAY: Record<TrafficEventType, string> = {
  sports:  '22 km radius',
  concert: '22 km radius',
  festival:'28 km radius',
  rally:   '32 km radius',
};

function fmtAttendees(n: number) {
  return n >= 1000 ? `${Math.round(n / 1000)}k` : String(n);
}

function CustomEventsSection() {
  const [showForm, setShowForm]             = useState(false);
  const [pendingName, setPendingName]       = useState('');
  const [pendingAttendees, setPendingAttendees] = useState(25000);
  const [pendingType, setPendingType]       = useState<TrafficEventType>('sports');

  const customEvents           = useSimulationStore((s) => s.customEvents);
  const placingEvent           = useSimulationStore((s) => s.placingEvent);
  const pendingEventLocation   = useSimulationStore((s) => s.pendingEventLocation);
  const addCustomEvent         = useSimulationStore((s) => s.addCustomEvent);
  const removeCustomEvent      = useSimulationStore((s) => s.removeCustomEvent);
  const setPlacingEvent        = useSimulationStore((s) => s.setPlacingEvent);
  const setPendingEventLocation= useSimulationStore((s) => s.setPendingEventLocation);

  const resetForm = () => {
    setShowForm(false);
    setPlacingEvent(false);
    setPendingName('');
    setPendingAttendees(25000);
    setPendingType('sports');
  };

  const handleConfirm = () => {
    if (!pendingEventLocation) return;
    addCustomEvent({
      id: `evt-${Date.now()}`,
      name: pendingName.trim() || `${pendingType.charAt(0).toUpperCase() + pendingType.slice(1)} Event`,
      lng: pendingEventLocation.lng,
      lat: pendingEventLocation.lat,
      attendees: pendingAttendees,
      type: pendingType,
    });
    resetForm();
  };

  const sliderPct = ((pendingAttendees - 5000) / 95000) * 100;

  return (
    <Section title="Predictive Traffic Events">
      <div className="space-y-1.5">
        {/* Existing events list */}
        {customEvents.map((event) => (
          <div
            key={event.id}
            className="flex items-start gap-2 px-2.5 py-2 rounded-lg bg-surface-800/40 border border-surface-700/30"
          >
            <div
              className="w-2 h-2 rounded-full flex-shrink-0 mt-0.5"
              style={{ backgroundColor: EVENT_TYPE_COLORS[event.type] }}
            />
            <div className="flex-1 min-w-0">
              <div className="text-[11px] text-slate-300 font-medium truncate">{event.name}</div>
              <div className="text-[10px] text-surface-500 mt-0.5">
                {fmtAttendees(event.attendees)} attendees · {event.type}
              </div>
              <div className="text-[10px] mt-0.5" style={{ color: EVENT_TYPE_COLORS[event.type] }}>
                {EVENT_TYPE_DECAY[event.type]} impact zone
              </div>
            </div>
            <button
              onClick={() => removeCustomEvent(event.id)}
              className="text-surface-600 hover:text-slate-400 text-[10px] flex-shrink-0 transition-colors mt-0.5"
            >✕</button>
          </div>
        ))}

        {/* Add event form */}
        {showForm ? (
          <div className="rounded-lg border border-surface-700/40 bg-surface-800/25 p-2.5 space-y-2.5">

            {/* Name */}
            <input
              type="text"
              placeholder="Event name (optional)"
              value={pendingName}
              onChange={(e) => setPendingName(e.target.value)}
              className="w-full bg-surface-900/60 border border-surface-700/40 rounded-md px-2.5 py-1.5 text-[11px] text-slate-300 placeholder-surface-600 outline-none focus:border-surface-500/60 transition-colors"
            />

            {/* Attendees */}
            <div>
              <div className="flex justify-between mb-1.5">
                <span className="text-[10px] text-surface-500">Attendees</span>
                <span className="text-[12px] font-mono font-bold text-slate-200">
                  {fmtAttendees(pendingAttendees)}
                </span>
              </div>
              <input
                type="range"
                min={5000}
                max={100000}
                step={1000}
                value={pendingAttendees}
                onChange={(e) => setPendingAttendees(parseInt(e.target.value))}
                className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, ${EVENT_TYPE_COLORS[pendingType]} ${sliderPct}%, #1e293b ${sliderPct}%)`,
                }}
              />
              <div className="flex justify-between text-[10px] text-surface-600 mt-0.5">
                <span>5k</span><span>100k</span>
              </div>
            </div>

            {/* Event type */}
            <div>
              <div className="text-[10px] text-surface-500 mb-1.5">Event type</div>
              <div className="grid grid-cols-2 gap-1">
                {(['sports', 'concert', 'festival', 'rally'] as TrafficEventType[]).map((type) => (
                  <button
                    key={type}
                    onClick={() => setPendingType(type)}
                    className="py-1.5 rounded-md text-[10px] font-medium border transition-all capitalize"
                    style={
                      pendingType === type
                        ? { borderColor: EVENT_TYPE_COLORS[type], color: EVENT_TYPE_COLORS[type], background: `${EVENT_TYPE_COLORS[type]}18` }
                        : { borderColor: '#1e293b', color: '#475569' }
                    }
                  >{type}</button>
                ))}
              </div>
              <div className="text-[10px] text-surface-600 mt-1.5">
                Traffic spread: <span className="text-surface-500">{EVENT_TYPE_DECAY[pendingType]}</span>
              </div>
            </div>

            {/* Location placement */}
            {!pendingEventLocation ? (
              <button
                onClick={() => setPlacingEvent(true)}
                className={`w-full py-2 rounded-md text-[11px] font-medium border transition-all ${
                  placingEvent
                    ? 'border-amber-700/70 text-amber-400 bg-amber-950/30'
                    : 'border-surface-700/40 text-surface-400 hover:text-slate-300 hover:border-surface-600/60'
                }`}
              >
                {placingEvent ? '↗ Click the map to place...' : '📍 Place on map'}
              </button>
            ) : (
              <div className="flex items-center gap-2 py-1.5 px-2.5 rounded-md bg-surface-700/25 border border-surface-600/40">
                <span className="text-[10px] text-emerald-400">✓ Placed</span>
                <span className="text-[10px] text-surface-500 font-mono flex-1">
                  {pendingEventLocation.lat.toFixed(3)}, {pendingEventLocation.lng.toFixed(3)}
                </span>
                <button
                  onClick={() => { setPendingEventLocation(null); setPlacingEvent(true); }}
                  className="text-[10px] text-surface-600 hover:text-slate-400 transition-colors"
                >re-place</button>
              </div>
            )}

            {/* Confirm / Cancel */}
            <div className="flex gap-1.5">
              <button
                onClick={handleConfirm}
                disabled={!pendingEventLocation}
                className="flex-1 py-2 rounded-md text-[11px] font-semibold border transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                style={{ borderColor: '#164e63', color: '#22d3ee' }}
                onMouseEnter={(e) => { if (pendingEventLocation) (e.target as HTMLElement).style.background = 'rgba(8,145,178,0.12)'; }}
                onMouseLeave={(e) => { (e.target as HTMLElement).style.background = 'transparent'; }}
              >
                Simulate Impact
              </button>
              <button
                onClick={resetForm}
                className="px-3 py-2 rounded-md text-[11px] border border-surface-700/40 text-surface-500 hover:text-slate-400 transition-colors"
              >Cancel</button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => setShowForm(true)}
            className="w-full py-2 rounded-lg border border-dashed border-surface-700/50 text-[11px] text-surface-500 hover:text-slate-400 hover:border-surface-600/60 transition-all"
          >+ Add Traffic Event</button>
        )}
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

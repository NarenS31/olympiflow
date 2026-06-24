import { useSimulationStore } from '../../stores/simulationStore';

function formatHour(h: number): string {
  const hour = Math.floor(h) % 24;
  const min  = Math.round((h % 1) * 60);
  const hh   = String(hour).padStart(2, '0');
  const mm   = String(min).padStart(2, '0');
  return `${hh}:${mm}`;
}

const PEAK_START = 16;
const PEAK_END   = 20;

const EVENT_MARKERS: { hour: number; label: string; isPeak?: boolean }[] = [
  { hour: 9,  label: 'Morning' },
  { hour: 14, label: 'Midday'  },
  { hour: 17, label: 'Peak Collision',  isPeak: true },
  { hour: 20, label: 'Evening' },
];

const TICK_HOURS = [6, 9, 12, 15, 18, 21, 24];

export function TimelineSlider() {
  const timeOfDay      = useSimulationStore((s) => s.timeOfDay);
  const setTimeOfDay   = useSimulationStore((s) => s.setTimeOfDay);
  const isPlaying      = useSimulationStore((s) => s.isPlaying);
  const togglePlay     = useSimulationStore((s) => s.togglePlay);
  const playbackSpeed  = useSimulationStore((s) => s.playbackSpeed);
  const setPlaybackSpeed = useSimulationStore((s) => s.setPlaybackSpeed);
  const mode           = useSimulationStore((s) => s.mode);

  const modeLabel = mode === 'crisis' ? 'Crisis' : mode === 'event' ? 'Event Day' : 'Baseline';
  const isPeak    = timeOfDay >= PEAK_START && timeOfDay < PEAK_END;

  const peakStartPct = (PEAK_START / 24) * 100;
  const peakEndPct   = (PEAK_END   / 24) * 100;
  const progressPct  = (timeOfDay  / 24) * 100;

  return (
    <div
      className="fixed z-40"
      style={{
        bottom: '30px',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '680px',
        maxWidth: 'calc(100vw - 2rem)',
        background: '#1C1C1E',
        border: '0.5px solid rgba(255,255,255,0.10)',
        borderRadius: '16px',
        padding: '10px 16px 14px',
        boxShadow: '0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.06)',
      }}
    >
      {/* Event marker labels */}
      <div className="relative mb-4" style={{ height: '18px' }}>
        {EVENT_MARKERS.map((m) => {
          const pct      = (m.hour / 24) * 100;
          const isActive = Math.abs(timeOfDay - m.hour) < 0.5;
          return (
            <div
              key={m.hour}
              className="absolute flex flex-col items-center"
              style={{ left: `${pct}%`, transform: 'translateX(-50%)', top: 0 }}
            >
              <div style={{
                width: '1px', height: '3px',
                background: isActive
                  ? (m.isPeak ? '#FF453A' : 'rgba(255,255,255,0.4)')
                  : 'rgba(255,255,255,0.12)',
                marginBottom: '2px',
              }} />
              <div style={{
                fontSize: m.isPeak ? '9px' : '9px',
                fontWeight: m.isPeak ? 600 : 400,
                color: m.isPeak
                  ? (isActive ? '#FF453A' : 'rgba(255,69,58,0.5)')
                  : (isActive ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.25)'),
                letterSpacing: '0.02em',
                whiteSpace: 'nowrap',
                transition: 'color 0.2s',
              }}>
                {m.label}
              </div>
            </div>
          );
        })}
      </div>

      {/* Main row */}
      <div className="flex items-center gap-3">
        {/* Play/Pause */}
        <button
          onClick={togglePlay}
          className="flex-shrink-0 flex items-center justify-center rounded-full transition-all"
          style={{
            width: '32px', height: '32px',
            background: isPlaying ? 'rgba(10,132,255,0.15)' : 'rgba(255,255,255,0.06)',
            border: 'none',
            color: isPlaying ? '#0A84FF' : 'rgba(255,255,255,0.5)',
            cursor: 'pointer',
          }}
        >
          {isPlaying ? (
            <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
            </svg>
          ) : (
            <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
          )}
        </button>

        {/* Time display */}
        <div className="flex-shrink-0 flex items-baseline gap-1.5" style={{ minWidth: '68px' }}>
          <span style={{
            fontSize: '18px',
            fontWeight: 500,
            fontFamily: "'SF Mono', ui-monospace, monospace",
            color: isPeak ? '#FF453A' : 'rgba(255,255,255,0.9)',
            lineHeight: 1,
            letterSpacing: '-0.02em',
          }}>
            {formatHour(timeOfDay)}
          </span>
          {isPlaying && (
            <span
              className="animate-cursor-blink"
              style={{ fontFamily: "'SF Mono', ui-monospace, monospace", fontSize: '16px', color: isPeak ? '#FF453A' : '#0A84FF', lineHeight: 1 }}
            >
              _
            </span>
          )}
          <span style={{
            fontSize: '9px',
            fontWeight: 500,
            color: isPeak ? 'rgba(255,69,58,0.6)' : 'rgba(255,255,255,0.25)',
            letterSpacing: '0.04em',
          }}>
            {isPeak ? 'PEAK' : modeLabel.toUpperCase()}
          </span>
        </div>

        {/* Slider */}
        <div className="flex-1 relative" style={{ paddingBottom: '16px' }}>
          {/* Tick labels */}
          <div className="flex justify-between absolute w-full" style={{ bottom: 0 }}>
            {TICK_HOURS.map((h) => (
              <span key={h} style={{
                fontSize: '9px',
                color: 'rgba(255,255,255,0.2)',
                fontFamily: "'SF Mono', ui-monospace, monospace",
                transform: 'translateX(-50%)',
              }}>
                {h === 24 ? '00' : String(h).padStart(2, '0')}
              </span>
            ))}
          </div>

          {/* Peak zone highlight */}
          {mode !== 'baseline' && (
            <div
              className="absolute pointer-events-none"
              style={{
                left: `${peakStartPct}%`,
                width: `${peakEndPct - peakStartPct}%`,
                top: '-12px',
                height: '2px',
                background: 'rgba(255,69,58,0.3)',
                borderRadius: '1px',
              }}
            />
          )}

          {/* Track bg */}
          <div
            className="absolute rounded-full pointer-events-none"
            style={{ height: '4px', top: '50%', transform: 'translateY(-50%)', left: 0, right: 0, background: 'rgba(255,255,255,0.08)' }}
          />

          {/* Progress fill */}
          <div
            className="absolute rounded-full pointer-events-none"
            style={{
              height: '4px', top: '50%', transform: 'translateY(-50%)',
              left: 0, width: `${progressPct}%`,
              background: isPeak ? 'linear-gradient(to right, #0A84FF, #FF453A)' : '#0A84FF',
            }}
          />

          <input
            type="range"
            min={0} max={24} step={0.1}
            value={timeOfDay}
            onChange={(e) => setTimeOfDay(parseFloat(e.target.value))}
            className="w-full appearance-none cursor-pointer relative"
            style={{ height: '4px', background: 'transparent', zIndex: 1 }}
          />
        </div>

        {/* Speed controls */}
        <div className="flex items-center gap-0.5 flex-shrink-0">
          {[1, 2, 5].map((s) => {
            const active = playbackSpeed === s;
            return (
              <button
                key={s}
                onClick={() => setPlaybackSpeed(s)}
                className="rounded-lg transition-all"
                style={{
                  padding: '4px 8px',
                  fontSize: '12px',
                  fontFamily: "'SF Mono', ui-monospace, monospace",
                  fontWeight: active ? 600 : 400,
                  color: active ? '#0A84FF' : 'rgba(255,255,255,0.3)',
                  background: active ? 'rgba(10,132,255,0.12)' : 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  letterSpacing: '-0.01em',
                }}
              >
                {s}×
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

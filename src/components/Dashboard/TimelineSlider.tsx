import { useSimulationStore } from '../../stores/simulationStore';

function formatHour(h: number): string {
  const hour = Math.floor(h) % 24;
  const min = Math.round((h % 1) * 60);
  const ampm = hour < 12 ? 'AM' : 'PM';
  const displayHour = hour % 12 || 12;
  return `${displayHour}:${String(min).padStart(2, '0')} ${ampm}`;
}

export function TimelineSlider() {
  const timeOfDay = useSimulationStore((s) => s.timeOfDay);
  const setTimeOfDay = useSimulationStore((s) => s.setTimeOfDay);
  const isPlaying = useSimulationStore((s) => s.isPlaying);
  const togglePlay = useSimulationStore((s) => s.togglePlay);
  const playbackSpeed = useSimulationStore((s) => s.playbackSpeed);
  const setPlaybackSpeed = useSimulationStore((s) => s.setPlaybackSpeed);

  const TICK_HOURS = [6, 9, 12, 15, 18, 21, 24];

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-40 w-[640px] max-w-[calc(100vw-2rem)] bg-surface-900/90 backdrop-blur-xl border border-surface-700/50 rounded-xl px-5 py-3 shadow-2xl">
      <div className="flex items-center gap-4">
        {/* Play/Pause */}
        <button
          onClick={togglePlay}
          className="w-9 h-9 flex items-center justify-center rounded-full bg-surface-700/60 border border-surface-600/60 text-slate-400 hover:text-slate-200 hover:bg-surface-600/60 transition-all flex-shrink-0"
        >
          {isPlaying ? (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
            </svg>
          ) : (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
          )}
        </button>

        {/* Time display */}
        <div className="text-[14px] font-mono font-bold text-slate-300 w-20 flex-shrink-0">
          {formatHour(timeOfDay)}
        </div>

        {/* Slider */}
        <div className="flex-1 relative">
          {/* Tick marks */}
          <div className="flex justify-between absolute -top-4 left-0 right-0">
            {TICK_HOURS.map((h) => (
              <span key={h} className="text-[9px] text-surface-600 font-mono">
                {h === 24 ? '12am' : h < 12 ? `${h}am` : h === 12 ? '12pm' : `${h - 12}pm`}
              </span>
            ))}
          </div>
          <input
            type="range"
            min={0}
            max={24}
            step={0.1}
            value={timeOfDay}
            onChange={(e) => setTimeOfDay(parseFloat(e.target.value))}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, #475569 ${(timeOfDay / 24) * 100}%, #1e293b ${(timeOfDay / 24) * 100}%)`,
            }}
          />
        </div>

        {/* Speed control */}
        <div className="flex items-center gap-1 flex-shrink-0">
          {[1, 2, 5].map((s) => (
            <button
              key={s}
              onClick={() => setPlaybackSpeed(s)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono transition-all ${
                playbackSpeed === s
                  ? 'bg-surface-600 text-slate-200 border border-surface-500'
                  : 'text-surface-600 hover:text-slate-400'
              }`}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

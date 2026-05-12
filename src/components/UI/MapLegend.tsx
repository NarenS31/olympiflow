import { useState } from 'react';

const VENUE_TYPES = [
  {
    color: '#b45309',
    bg: 'rgba(180,83,9,0.10)',
    label: 'Opening / Closing Ceremony',
    icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C9.5 5.5 8 8.5 9 11.5c.5 1.5 1.5 2.5 2.5 3C11 13 11.5 11.5 13 11c-.5 3 1 5 2 6.5C16 15.5 16 13 14.5 11c1.5 1 2.5 3 2.5 5 1-1.5 1.5-3.5 1-5.5C17 7 14.5 4 12 2z"/></svg>`,
  },
  {
    color: '#0891b2',
    bg: 'rgba(8,145,178,0.08)',
    label: 'Stadium',
    icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 11C3 7 7 4 12 4s9 3 9 7"/><path d="M3 11v6h18v-6"/><line x1="7" y1="17" x2="7" y2="11"/><line x1="17" y1="17" x2="17" y2="11"/><line x1="12" y1="17" x2="12" y2="11"/></svg>`,
  },
  {
    color: '#7c3aed',
    bg: 'rgba(124,58,237,0.08)',
    label: 'Arena / Indoor Venue',
    icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M2 13 Q12 3 22 13"/><rect x="2" y="13" width="20" height="5" rx="1"/><line x1="7" y1="18" x2="7" y2="13"/><line x1="17" y1="18" x2="17" y2="13"/></svg>`,
  },
  {
    color: '#16a34a',
    bg: 'rgba(22,163,74,0.08)',
    label: 'Outdoor / Park Venue',
    icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 3 16 21 16"/><line x1="12" y1="16" x2="12" y2="22"/></svg>`,
  },
  {
    color: '#1d4ed8',
    bg: 'rgba(29,78,216,0.08)',
    label: 'Aquatic Venue',
    icon: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M2 12 Q5.5 7 9 12 Q12.5 17 16 12 Q19.5 7 23 12"/><path d="M2 17 Q5.5 12 9 17 Q12.5 22 16 17 Q19.5 12 23 17"/></svg>`,
  },
];

const ZONE_LEVELS = [
  { color: '#166534', label: 'Low' },
  { color: '#854d0e', label: 'Moderate' },
  { color: '#9a3412', label: 'Heavy' },
  { color: '#991b1b', label: 'Critical' },
];

export function MapLegend() {
  const [open, setOpen] = useState(false);

  return (
    <div className="fixed bottom-4 right-4 z-40 flex flex-col items-end gap-2">
      {open && (
        <div className="w-52 bg-surface-900/95 backdrop-blur-xl border border-surface-700/50 rounded-xl overflow-hidden shadow-2xl">
          {/* Venue types */}
          <div className="px-3 pt-3 pb-2">
            <div className="text-[10px] font-semibold text-surface-600 uppercase tracking-widest mb-2">
              Venue Types
            </div>
            <div className="space-y-1.5">
              {VENUE_TYPES.map((t) => (
                <div key={t.label} className="flex items-center gap-2">
                  <div className="flex flex-col items-center flex-shrink-0">
                    <div
                      className="w-5 h-5 rounded-full flex items-center justify-center"
                      style={{ border: `1.5px solid ${t.color}`, background: t.bg, color: t.color }}
                      dangerouslySetInnerHTML={{ __html: t.icon }}
                    />
                    <div
                      className="w-0 h-0"
                      style={{
                        borderLeft: '3px solid transparent',
                        borderRight: '3px solid transparent',
                        borderTop: `5px solid ${t.color}`,
                        marginTop: '-1px',
                      }}
                    />
                  </div>
                  <span className="text-[11px] text-surface-400 leading-tight">{t.label}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="border-t border-surface-700/50 mx-3" />

          {/* Zone overlay */}
          <div className="px-3 pt-2 pb-3">
            <div className="text-[10px] font-semibold text-surface-600 uppercase tracking-widest mb-2">
              Zone Congestion
            </div>
            <div className="flex items-center gap-1 mb-1">
              {ZONE_LEVELS.map((z) => (
                <div
                  key={z.color}
                  className="flex-1 h-2 rounded-sm"
                  style={{ background: z.color, opacity: 0.75 }}
                />
              ))}
            </div>
            <div className="flex justify-between">
              <span className="text-[9px] text-surface-600">Low</span>
              <span className="text-[9px] text-surface-600">Critical</span>
            </div>
          </div>
        </div>
      )}

      {/* Toggle button */}
      <button
        onClick={() => setOpen((v) => !v)}
        className={`w-9 h-9 flex items-center justify-center rounded-lg border text-surface-400 hover:text-slate-200 transition-all ${
          open
            ? 'bg-surface-700/80 border-surface-600 text-slate-200'
            : 'bg-surface-900/90 border-surface-700/50 hover:bg-surface-800/80'
        }`}
        title="Map legend"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
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

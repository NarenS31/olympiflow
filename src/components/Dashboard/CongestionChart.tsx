import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { useSimulationStore } from '../../stores/simulationStore';
import { generateTimelineData } from '../../utils/simulation';

interface TooltipProps {
  active?: boolean;
  payload?: { value: number; name: string; color: string }[];
  label?: string;
}

function CustomTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(17,19,24,0.95)',
      border: '0.5px solid rgba(255,255,255,0.1)',
      borderRadius: '8px',
      padding: '8px 12px',
      backdropFilter: 'blur(20px)',
    }}>
      <div style={{ color: 'rgba(255,255,255,0.3)', marginBottom: '4px', fontSize: '10px', fontFamily: "'SF Mono', ui-monospace, monospace" }}>
        {label}
      </div>
      {payload.map((p) => (
        <div key={p.name} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: p.color }} />
          <span style={{ color: 'rgba(255,255,255,0.4)', fontSize: '11px', textTransform: 'capitalize' }}>{p.name}</span>
          <span style={{ fontFamily: "'SF Mono', ui-monospace, monospace", fontWeight: 600, color: p.color, fontSize: '12px' }}>
            {Math.round(p.value * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export function CongestionChart() {
  const globalIntensity = useSimulationStore((s) => s.globalIntensity);
  const venueSurges = useSimulationStore((s) => s.venueSurges);
  const data = generateTimelineData(globalIntensity, venueSurges);

  return (
    <div className="h-full w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 4, left: -24, bottom: 0 }}>
          <defs>
            <linearGradient id="simGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#0A84FF" stopOpacity={0.18} />
              <stop offset="100%" stopColor="#0A84FF" stopOpacity={0.01} />
            </linearGradient>
            <linearGradient id="baseGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="rgba(255,255,255,0.12)" stopOpacity={0.6} />
              <stop offset="100%" stopColor="rgba(255,255,255,0.12)" stopOpacity={0.01} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="hour"
            tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 9, fontFamily: "'SF Mono', ui-monospace, monospace" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: string) => {
              const h = parseInt(v);
              if (isNaN(h)) return v;
              return h < 12 ? `${h}A` : h === 12 ? '12P' : `${h-12}P`;
            }}
          />
          <YAxis
            tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 9, fontFamily: "'SF Mono', ui-monospace, monospace" }}
            axisLine={false}
            tickLine={false}
            domain={[0, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}`}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            y={0.60}
            stroke="rgba(255,69,58,0.6)"
            strokeDasharray="4 3"
            strokeWidth={0.5}
          />
          <Area
            type="monotone"
            dataKey="baseline"
            stroke="rgba(255,255,255,0.15)"
            strokeWidth={1}
            fill="url(#baseGrad)"
            name="Baseline"
            strokeDasharray="4 3"
          />
          <Area
            type="monotone"
            dataKey="congestion"
            stroke="#0A84FF"
            strokeWidth={1.5}
            fill="url(#simGrad)"
            name="Simulated"
            fillOpacity={1}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

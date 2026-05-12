import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
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
    <div className="bg-surface-800 border border-surface-700/60 rounded-lg px-3 py-2 text-[11px]">
      <div className="text-surface-500 mb-1">{label}</div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }} />
          <span className="text-surface-500 capitalize">{p.name}:</span>
          <span className="font-mono font-bold" style={{ color: p.color }}>
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
        <AreaChart data={data} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
          <defs>
            <linearGradient id="simGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#9a3412" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#9a3412" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="baseGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#334155" stopOpacity={0.5} />
              <stop offset="95%" stopColor="#334155" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a2332" vertical={false} />
          <XAxis
            dataKey="hour"
            tick={{ fill: '#3d5068', fontSize: 9 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#3d5068', fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            domain={[0, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="baseline"
            stroke="#334155"
            strokeWidth={1.5}
            fill="url(#baseGrad)"
            name="Baseline"
            strokeDasharray="4 2"
          />
          <Area
            type="monotone"
            dataKey="congestion"
            stroke="#9a3412"
            strokeWidth={1.5}
            fill="url(#simGrad)"
            name="Simulated"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

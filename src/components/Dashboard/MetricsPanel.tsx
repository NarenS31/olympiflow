import { useSimulationStore } from '../../stores/simulationStore';
import { CongestionChart } from './CongestionChart';

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  color: string;
  trend?: 'up' | 'down' | 'neutral';
}

function MetricCard({ label, value, sub, color, trend }: MetricCardProps) {
  return (
    <div className="bg-surface-800/40 rounded-lg px-3 py-2.5 border border-surface-700/30">
      <div className="flex items-start justify-between gap-1">
        <div>
          <div className="text-[10px] text-surface-500 uppercase tracking-wider mb-1">{label}</div>
          <div className={`text-[18px] font-bold font-mono leading-none ${color}`}>{value}</div>
          {sub && <div className="text-[10px] text-surface-600 mt-0.5">{sub}</div>}
        </div>
        {trend && trend !== 'neutral' && (
          <span className={`text-[11px] font-mono mt-1 ${
            trend === 'up' ? 'text-red-500' : 'text-slate-500'
          }`}>
            {trend === 'up' ? '▲' : '▼'}
          </span>
        )}
      </div>
    </div>
  );
}

export function MetricsPanel() {
  const metrics = useSimulationStore((s) => s.metrics);
  const mode = useSimulationStore((s) => s.mode);

  const delayColor =
    metrics.avgDelayIncrease > 60 ? 'text-red-400' :
    metrics.avgDelayIncrease > 30 ? 'text-amber-500' :
    'text-slate-300';

  const congColor =
    metrics.congestionScore > 0.7 ? 'text-red-400' :
    metrics.congestionScore > 0.4 ? 'text-amber-500' :
    'text-slate-300';

  const statusDot =
    mode === 'crisis' ? 'bg-red-600' :
    mode === 'event' ? 'bg-amber-700' :
    'bg-slate-600';

  const statusLabel =
    mode === 'crisis' ? 'Crisis Mode Active' :
    mode === 'event' ? 'Olympic Event Day' :
    'Baseline — Normal Traffic';

  return (
    <div className="fixed right-4 top-[68px] bottom-[100px] z-40 w-64 flex flex-col gap-3">
      {/* Status badge */}
      <div className="flex items-center gap-2 bg-surface-900/90 backdrop-blur-xl border border-surface-700/50 rounded-xl px-3 py-2">
        <div className={`w-2 h-2 rounded-full flex-shrink-0 ${statusDot}`} />
        <span className="text-[11px] font-medium text-surface-400 uppercase tracking-wider">
          {statusLabel}
        </span>
      </div>

      {/* Metric cards */}
      <div className="bg-surface-900/90 backdrop-blur-xl border border-surface-700/50 rounded-xl p-3 space-y-2">
        <div className="text-[10px] font-semibold text-surface-600 uppercase tracking-widest mb-1">
          Live Metrics
        </div>
        <MetricCard
          label="Avg. Delay"
          value={`+${metrics.avgDelayIncrease}%`}
          sub="vs. free-flow travel"
          color={delayColor}
          trend={metrics.avgDelayIncrease > 20 ? 'up' : 'neutral'}
        />
        <MetricCard
          label="Congestion Score"
          value={`${Math.round(metrics.congestionScore * 100)}%`}
          sub="network pressure"
          color={congColor}
          trend={metrics.congestionScore > 0.4 ? 'up' : 'neutral'}
        />
        <MetricCard
          label="Peak Zones"
          value={String(metrics.peakCongestionZones)}
          sub="high-pressure areas"
          color="text-slate-300"
        />
        <MetricCard
          label="Affected Routes"
          value={String(metrics.affectedTransitRoutes)}
          sub="transit disruptions"
          color="text-slate-300"
          trend={metrics.affectedTransitRoutes > 10 ? 'up' : 'neutral'}
        />
        <MetricCard
          label="Persons Affected"
          value={
            metrics.estimatedPersonsAffected > 0
              ? `${(metrics.estimatedPersonsAffected / 1000).toFixed(0)}K`
              : '—'
          }
          sub="estimated commuters"
          color={metrics.estimatedPersonsAffected > 50000 ? 'text-amber-500' : 'text-slate-300'}
        />
      </div>

      {/* Chart */}
      <div className="bg-surface-900/90 backdrop-blur-xl border border-surface-700/50 rounded-xl p-3 flex-1 min-h-0">
        <div className="text-[10px] font-semibold text-surface-600 uppercase tracking-widest mb-2">
          Congestion Over Time
        </div>
        <div className="h-full pb-4">
          <CongestionChart />
        </div>
      </div>
    </div>
  );
}

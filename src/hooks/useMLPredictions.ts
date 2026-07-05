import { useEffect, useCallback, useRef } from 'react';
import { useSimulationStore } from '../stores/simulationStore';
import { fetchMLSurgePrediction, fetchMLConditionClassification } from '../api/client';

// Refresh ML predictions every 8 seconds while the sim runs, or on key state changes
const REFRESH_INTERVAL_MS = 8000;

export function useMLPredictions() {
  const mode            = useSimulationStore((s) => s.mode);
  const timeOfDay       = useSimulationStore((s) => s.timeOfDay);
  const globalIntensity = useSimulationStore((s) => s.globalIntensity);
  const venueSurges     = useSimulationStore((s) => s.venueSurges);
  const staggered       = useSimulationStore((s) => s.staggeredArrivals);
  const setML           = useSimulationStore((s) => s.setMLPredictions);
  const intervalRef     = useRef<ReturnType<typeof setInterval> | null>(null);

  const runPredictions = useCallback(async () => {
    // Pick the most congested venue (or default to sofi)
    const surgeEntries = Object.entries(venueSurges);
    const topVenue = surgeEntries.length
      ? surgeEntries.reduce((a, b) => (b[1] > a[1] ? b : a))[0]
      : 'sofi';

    const eventType =
      mode === 'crisis' ? 2 : mode === 'event' ? 1 : 0;

    const rawVc = 0.3 + globalIntensity * 1.2;
    const vc = staggered ? Math.max(0.1, rawVc * 0.75) : rawVc;

    const concurrentEvents = surgeEntries.filter(([, v]) => v > 0.1).length;
    const transitScore = staggered ? 0.65 : 0.4;

    try {
      const [surgeRes, condRes] = await Promise.all([
        fetchMLSurgePrediction({
          venue_id: topVenue,
          hour: timeOfDay,
          day_of_week: new Date().getDay(),
          event_type: eventType,
          capacity_util: Math.min(1, globalIntensity),
        }),
        fetchMLConditionClassification({
          surge_intensity: globalIntensity,
          vc_ratio: vc,
          time_to_event_min: mode === 'event' ? 30 : mode === 'crisis' ? -10 : 120,
          concurrent_events: concurrentEvents,
          transit_availability_score: transitScore,
        }),
      ]);

      setML({
        surgeIntensity: surgeRes.intensity,
        confidence: surgeRes.confidence,
        condition: condRes.condition,
        probabilities: condRes.probabilities,
        loading: false,
      });
    } catch {
      // silently fail — ML card will stay at last known values
    }
  }, [mode, timeOfDay, globalIntensity, venueSurges, staggered, setML]);

  useEffect(() => {
    runPredictions();
  }, [runPredictions]);

  useEffect(() => {
    intervalRef.current = setInterval(runPredictions, REFRESH_INTERVAL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [runPredictions]);
}

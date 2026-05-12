import { useEffect, useRef, useCallback } from 'react';
import { useSimulationStore } from '../stores/simulationStore';
import { fetchHeatmapData, fetchTransitRoutes } from '../api/client';
import { calculateMetrics } from '../utils/simulation';

export function useSimulationData() {
  const setHeatmapBaseData = useSimulationStore((s) => s.setHeatmapBaseData);
  const setTransitData = useSimulationStore((s) => s.setTransitData);

  useEffect(() => {
    fetchHeatmapData()
      .then(setHeatmapBaseData)
      .catch(() => {
        // Backend not available; use empty fallback
        setHeatmapBaseData({ type: 'FeatureCollection', features: [] });
      });

    fetchTransitRoutes()
      .then(setTransitData)
      .catch(() => {
        setTransitData({ type: 'FeatureCollection', features: [] });
      });
  }, [setHeatmapBaseData, setTransitData]);
}

export function useSimulationTick() {
  const isPlaying = useSimulationStore((s) => s.isPlaying);
  const playbackSpeed = useSimulationStore((s) => s.playbackSpeed);
  const timeOfDay = useSimulationStore((s) => s.timeOfDay);
  const setTimeOfDay = useSimulationStore((s) => s.setTimeOfDay);
  const venueSurges = useSimulationStore((s) => s.venueSurges);
  const globalIntensity = useSimulationStore((s) => s.globalIntensity);
  const updateMetrics = useSimulationStore((s) => s.updateMetrics);

  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const recalcMetrics = useCallback(() => {
    const result = calculateMetrics(venueSurges, globalIntensity, timeOfDay);
    updateMetrics({
      congestionScore: result.congestionScore,
      avgDelayIncrease: result.avgDelayIncrease,
      peakCongestionZones: result.peakCongestionZones,
      affectedTransitRoutes: result.affectedTransitRoutes,
      estimatedPersonsAffected: result.estimatedPersonsAffected,
    });
  }, [venueSurges, globalIntensity, timeOfDay, updateMetrics]);

  // Recalculate metrics whenever simulation state changes
  useEffect(() => {
    recalcMetrics();
  }, [recalcMetrics]);

  // Advance time when playing
  useEffect(() => {
    if (tickRef.current) clearInterval(tickRef.current);

    if (isPlaying) {
      tickRef.current = setInterval(() => {
        setTimeOfDay((prev: number) => {
          const next = prev + 0.1 * playbackSpeed;
          return next >= 24 ? 0 : next;
        });
      }, 200);
    }

    return () => {
      if (tickRef.current) clearInterval(tickRef.current);
    };
  }, [isPlaying, playbackSpeed, setTimeOfDay]);
}

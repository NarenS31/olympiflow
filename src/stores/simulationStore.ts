import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { SimulationMode, LayerVisibility, SimulationMetrics, VenueSurge } from '../types';

interface SimulationState {
  mode: SimulationMode;
  timeOfDay: number;
  isPlaying: boolean;
  playbackSpeed: number;
  globalIntensity: number;
  venueSurges: Record<string, number>; // venueId -> intensity 0-1
  roadClosures: string[];
  layers: LayerVisibility;
  metrics: SimulationMetrics;
  selectedVenueId: string | null;
  transitData: GeoJSON.FeatureCollection | null;
  heatmapBaseData: GeoJSON.FeatureCollection | null;

  setMode: (mode: SimulationMode) => void;
  setTimeOfDay: (t: number | ((prev: number) => number)) => void;
  togglePlay: () => void;
  setPlaybackSpeed: (s: number) => void;
  setGlobalIntensity: (i: number) => void;
  setVenueSurge: (venueId: string, intensity: number) => void;
  removeVenueSurge: (venueId: string) => void;
  toggleLayer: (layer: keyof LayerVisibility) => void;
  selectVenue: (venueId: string | null) => void;
  setTransitData: (data: GeoJSON.FeatureCollection) => void;
  setHeatmapBaseData: (data: GeoJSON.FeatureCollection) => void;
  updateMetrics: (metrics: Partial<SimulationMetrics>) => void;
  resetSimulation: () => void;
}

const DEFAULT_METRICS: SimulationMetrics = {
  avgDelayIncrease: 0,
  peakCongestionZones: 0,
  affectedTransitRoutes: 0,
  estimatedPersonsAffected: 0,
  congestionScore: 0,
};

export const useSimulationStore = create<SimulationState>()(
  subscribeWithSelector((set) => ({
    mode: 'baseline',
    timeOfDay: 8,
    isPlaying: false,
    playbackSpeed: 1,
    globalIntensity: 0.3,
    venueSurges: {},
    roadClosures: [],
    layers: {
      heatmap: true,
      venues: true,
      transit: true,
      parking: false,
      collisions: false,
    },
    metrics: DEFAULT_METRICS,
    selectedVenueId: null,
    transitData: null,
    heatmapBaseData: null,

    setMode: (mode) =>
      set((state) => {
        const updates: Partial<SimulationState> = { mode };
        if (mode === 'baseline') {
          updates.globalIntensity = 0.3;
          updates.venueSurges = {};
        } else if (mode === 'event') {
          updates.globalIntensity = 0.65;
        } else if (mode === 'crisis') {
          updates.globalIntensity = 1.0;
        }
        return updates;
      }),

    setTimeOfDay: (t) =>
      set((s) => ({ timeOfDay: typeof t === 'function' ? t(s.timeOfDay) : t })),

    togglePlay: () => set((s) => ({ isPlaying: !s.isPlaying })),

    setPlaybackSpeed: (s) => set({ playbackSpeed: s }),

    setGlobalIntensity: (i) => set({ globalIntensity: i }),

    setVenueSurge: (venueId, intensity) =>
      set((s) => ({ venueSurges: { ...s.venueSurges, [venueId]: intensity } })),

    removeVenueSurge: (venueId) =>
      set((s) => {
        const next = { ...s.venueSurges };
        delete next[venueId];
        return { venueSurges: next };
      }),

    toggleLayer: (layer) =>
      set((s) => ({ layers: { ...s.layers, [layer]: !s.layers[layer] } })),

    selectVenue: (venueId) => set({ selectedVenueId: venueId }),

    setTransitData: (data) => set({ transitData: data }),

    setHeatmapBaseData: (data) => set({ heatmapBaseData: data }),

    updateMetrics: (metrics) =>
      set((s) => ({ metrics: { ...s.metrics, ...metrics } })),

    resetSimulation: () =>
      set({
        mode: 'baseline',
        globalIntensity: 0.3,
        venueSurges: {},
        roadClosures: [],
        metrics: DEFAULT_METRICS,
        isPlaying: false,
      }),
  }))
);

export type { VenueSurge };

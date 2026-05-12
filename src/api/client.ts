import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 15000,
});

export async function fetchHeatmapData(): Promise<GeoJSON.FeatureCollection> {
  const { data } = await api.get('/traffic/heatmap');
  return data;
}

export async function fetchTransitRoutes(): Promise<GeoJSON.FeatureCollection> {
  const { data } = await api.get('/transit/routes');
  return data;
}

export async function fetchVenues(): Promise<GeoJSON.FeatureCollection> {
  const { data } = await api.get('/venues/geojson');
  return data;
}

export async function fetchTrafficStats() {
  const { data } = await api.get('/traffic/stats');
  return data as {
    topIntersections: { street: string; total: number }[];
    totalCount: number;
    avgDailyVolume: number;
  };
}

export async function fetchParkingData(): Promise<GeoJSON.FeatureCollection> {
  const { data } = await api.get('/transit/parking');
  return data;
}

export async function fetchCrimeData(): Promise<GeoJSON.FeatureCollection> {
  const { data } = await api.get('/traffic/crime');
  return data;
}

export interface AIAdvisorResponse {
  answer: string;
  context_used: string[];
  model: string;
}

export async function askAIAdvisor(payload: {
  query: string;
  model: string;
  simulation_context?: Record<string, unknown>;
}): Promise<AIAdvisorResponse> {
  const { data } = await api.post('/ai/ask', payload);
  return data as AIAdvisorResponse;
}

export async function postSimulationStep(payload: {
  mode: string;
  timeOfDay: number;
  globalIntensity: number;
  venueSurges: Record<string, number>;
}) {
  const { data } = await api.post('/simulation/step', payload);
  return data as {
    congestionScore: number;
    avgDelayIncrease: number;
    peakZones: number;
    affectedRoutes: number;
    personsAffected: number;
  };
}

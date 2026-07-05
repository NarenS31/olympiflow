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
  const { data } = await api.post('/ai/ask', payload, { timeout: 120000 });
  return data as AIAdvisorResponse;
}

export interface MLSurgePrediction {
  intensity: number;
  confidence: number;
  venue_id: string;
  hour: number;
}

export interface MLConditionPrediction {
  condition: 'NORMAL' | 'ELEVATED' | 'PEAK' | 'CRITICAL';
  condition_index: number;
  probabilities: Record<string, number>;
}

export async function fetchMLSurgePrediction(payload: {
  venue_id: string;
  hour: number;
  day_of_week: number;
  event_type: number;
  capacity_util: number;
}): Promise<MLSurgePrediction> {
  const { data } = await api.post('/ml/predict-surge', payload);
  return data as MLSurgePrediction;
}

export async function fetchMLConditionClassification(payload: {
  surge_intensity: number;
  vc_ratio: number;
  time_to_event_min: number;
  concurrent_events: number;
  transit_availability_score: number;
}): Promise<MLConditionPrediction> {
  const { data } = await api.post('/ml/classify-conditions', payload);
  return data as MLConditionPrediction;
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

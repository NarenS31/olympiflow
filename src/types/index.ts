export interface Venue {
  id: string;
  name: string;
  shortName: string;
  sports: string[];
  lng: number;
  lat: number;
  capacity: number;
  type: 'stadium' | 'arena' | 'outdoor' | 'aquatic' | 'velodrome';
  neighborhood: string;
  isOpeningClosing?: boolean;
}

export interface HeatmapPoint {
  lng: number;
  lat: number;
  weight: number;
}

export interface TransitRoute {
  id: string;
  name: string;
  color: string;
  geometry: GeoJSON.LineString | GeoJSON.MultiLineString;
}

export interface ParkingZone {
  id: string;
  lng: number;
  lat: number;
  occupancyRate: number;
  totalSpaces: number;
}

export type SimulationMode = 'baseline' | 'event' | 'crisis';

export type TrafficEventType = 'sports' | 'concert' | 'festival' | 'rally';

export interface CustomTrafficEvent {
  id: string;
  name: string;
  lng: number;
  lat: number;
  attendees: number;
  type: TrafficEventType;
}

export interface VenueSurge {
  venueId: string;
  intensity: number; // 0-1
}

export interface LayerVisibility {
  heatmap: boolean;
  venues: boolean;
  transit: boolean;
  parking: boolean;
  collisions: boolean;
}

export type EmergencyRisk = 'low' | 'moderate' | 'high' | 'critical';

export interface SimulationMetrics {
  avgDelayIncrease: number;       // percent above free-flow
  peakCongestionZones: number;    // count
  affectedTransitRoutes: number;  // count
  estimatedPersonsAffected: number;
  congestionScore: number;        // 0-1
  avgTravelTime: number;          // minutes
  co2Increase: number;            // percent above baseline
  emergencyDelayRisk: EmergencyRisk;
}

export interface TimelineDataPoint {
  hour: string;
  congestion: number;
  baseline: number;
}

export interface GeoJSONFeatureCollection {
  type: 'FeatureCollection';
  features: GeoJSON.Feature[];
}

// GeoJSON types are provided globally by @types/geojson (via maplibre-gl)

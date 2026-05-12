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

export interface SimulationMetrics {
  avgDelayIncrease: number;
  peakCongestionZones: number;
  affectedTransitRoutes: number;
  estimatedPersonsAffected: number;
  congestionScore: number;
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

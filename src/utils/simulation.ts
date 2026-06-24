import { LA28_VENUES } from '../data/venues';
import { LA_ZONES } from '../data/zones';
import { haversineDistance } from './geoUtils';
import type { CustomTrafficEvent, TrafficEventType } from '../types';

// BPR (Bureau of Public Roads) travel-time function
// Returns travel time ratio: 1.0 = free flow, >1 = delayed
export function bprTravelTime(volume: number, capacity: number, alpha = 0.15, beta = 4): number {
  return 1 + alpha * Math.pow(volume / capacity, beta);
}

// Per-event-type time-of-day sensitivity — encodes historical arrival/departure patterns.
// Returns 0–1 multiplier: how intense traffic pressure is at a given hour for that event type.
export function getEventTimeSensitivity(type: TrafficEventType, hour: number): number {
  switch (type) {
    case 'sports':
      // Arrival spike ~2h before 7 PM start, departure spike ~1h after
      return Math.max(0.15, Math.exp(-Math.pow((hour - 18.5) / 2.2, 2)));
    case 'concert':
      // Tighter evening spike around 8 PM
      return Math.max(0.10, Math.exp(-Math.pow((hour - 20) / 1.8, 2)));
    case 'festival':
      // Broad afternoon-to-evening spread, 12–10 PM
      if (hour < 11) return 0.10;
      if (hour > 22) return 0.10;
      return 0.40 + 0.60 * Math.sin(((hour - 11) / 11) * Math.PI);
    case 'rally':
      // Midday concentrated burst
      return Math.max(0.10, Math.exp(-Math.pow((hour - 13) / 2.5, 2)));
  }
}

// Time-of-day traffic multiplier
export function getTimeMultiplier(hour: number): number {
  // Midnight -> 6am: night (low)
  if (hour < 6) return 0.2;
  // 6-9am: morning rush
  if (hour < 9) return 0.5 + ((hour - 6) / 3) * 0.5;
  // 9am-3pm: daytime
  if (hour < 15) return 0.6;
  // 3-7pm: evening rush
  if (hour < 19) return 0.7 + ((hour - 15) / 4) * 0.3;
  // 7-10pm: evening
  if (hour < 22) return 0.55;
  // 10pm+: late night
  return 0.25;
}

// Venue risk weights for heatmap — normalized from Olympic dataset risk scores.
// Coliseum (78.1) anchors at 1.0; everything else scales proportionally.
const VENUE_HEATMAP_WEIGHTS: Record<string, number> = {
  'la-coliseum':      1.00,
  'long-beach-arena': 0.45,
  'crypto-arena':     0.45,
  'sofi':             0.43,
  'rose-bowl':        0.43,
  'intuit-dome':      0.37,
  'bmo-stadium':      0.32,
  'pauley':           0.26,
  'dignity-health':   0.28,
  'sepulveda-basin':  0.20,
  'el-dorado':        0.19,
  'ucla-olympic':     0.26,
};

// Tight concentric pressure rings around a venue epicenter.
// Maximum radius ~5.3 km (0.048°) — stays venue-local, not city-wide.
function createVenueHotspot(
  lng: number,
  lat: number,
  peakWeight: number,
): { lng: number; lat: number; weight: number }[] {
  const pts: { lng: number; lat: number; weight: number }[] = [];
  const rings = [
    { r: 0.000, n:  1, w: 1.00 },
    { r: 0.005, n:  8, w: 0.88 },
    { r: 0.012, n: 12, w: 0.68 },
    { r: 0.022, n: 16, w: 0.42 },
    { r: 0.034, n: 16, w: 0.18 },
    { r: 0.048, n: 12, w: 0.06 },
  ];
  for (const ring of rings) {
    const w = ring.w * peakWeight;
    if (w < 0.02) continue;
    for (let i = 0; i < ring.n; i++) {
      const angle = (i / ring.n) * Math.PI * 2;
      pts.push({ lng: lng + ring.r * Math.cos(angle), lat: lat + ring.r * Math.sin(angle), weight: w });
    }
  }
  return pts;
}

// Diffuse congestion from a custom event epicenter in concentric rings (wider decay)
function createSurgePressurePoints(
  lng: number,
  lat: number,
  intensity: number,
): { lng: number; lat: number; weight: number }[] {
  const points: { lng: number; lat: number; weight: number }[] = [];
  const rings = [
    { radius: 0.000, rings: 1, weight: intensity },
    { radius: 0.008, rings: 8, weight: intensity * 0.85 },
    { radius: 0.018, rings: 12, weight: intensity * 0.65 },
    { radius: 0.032, rings: 16, weight: intensity * 0.4 },
    { radius: 0.055, rings: 20, weight: intensity * 0.2 },
    { radius: 0.085, rings: 24, weight: intensity * 0.08 },
  ];

  for (const ring of rings) {
    const count = ring.rings;
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * 2 * Math.PI;
      points.push({
        lng: lng + ring.radius * Math.cos(angle),
        lat: lat + ring.radius * Math.sin(angle),
        weight: ring.weight,
      });
    }
  }
  return points;
}

export function generateHeatmapGeoJSON(
  basePoints: { lng: number; lat: number; weight: number }[],
  venueSurges: Record<string, number>,
  globalIntensity: number,
  timeOfDay: number,
  customEvents: CustomTrafficEvent[] = [],
  staggeredArrivals = false,
): GeoJSON.FeatureCollection {
  const timeMult = getTimeMultiplier(timeOfDay);
  const staggerFactor = staggeredArrivals ? 0.68 : 1.0;
  const features: GeoJSON.Feature[] = [];

  // Base city road-density — extremely subtle, just hints at the road network.
  // Cap at 0.08 so base points never create city-wide red on their own.
  for (const p of basePoints) {
    const w = Math.min(0.08, p.weight * timeMult * globalIntensity * 0.10);
    if (w < 0.006) continue;
    features.push({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [p.lng, p.lat] },
      properties: { weight: w },
    });
  }

  // Per-venue hotspots — primary heatmap signal.
  // Each venue generates a tight cluster of points that fade to zero at ~5 km.
  for (const venue of LA28_VENUES) {
    const baseWeight = VENUE_HEATMAP_WEIGHTS[venue.id] ?? 0.20;
    const surgeBoost = 1.0 + (venueSurges[venue.id] ?? 0) * 0.8;
    const venueWeight = Math.min(1, baseWeight * globalIntensity * timeMult * surgeBoost * staggerFactor);
    if (venueWeight < 0.03) continue;
    for (const pt of createVenueHotspot(venue.lng, venue.lat, venueWeight)) {
      features.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [pt.lng, pt.lat] },
        properties: { weight: pt.weight },
      });
    }
  }

  // Custom event pressure rings
  for (const event of customEvents) {
    const normalizedIntensity = Math.min(1, event.attendees / 70000);
    const timeSensitivity = getEventTimeSensitivity(event.type, timeOfDay);
    const effectiveIntensity = normalizedIntensity * timeSensitivity * staggerFactor;
    if (effectiveIntensity < 0.05) continue;
    const surgePoints = createSurgePressurePoints(event.lng, event.lat, effectiveIntensity);
    for (const sp of surgePoints) {
      features.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [sp.lng, sp.lat] },
        properties: { weight: sp.weight },
      });
    }
  }

  return { type: 'FeatureCollection', features };
}

export function calculateMetrics(
  venueSurges: Record<string, number>,
  globalIntensity: number,
  timeOfDay: number,
) {
  const timeMult = getTimeMultiplier(timeOfDay);
  const surgeValues = Object.values(venueSurges);
  const totalSurge = surgeValues.reduce((a, b) => a + b, 0);
  const activeSurges = surgeValues.filter((v) => v > 0).length;

  const congestionScore = Math.min(
    1,
    globalIntensity * timeMult * 0.6 + (totalSurge / Math.max(1, activeSurges)) * 0.4,
  );

  const avgDelayIncrease = Math.round(bprTravelTime(congestionScore, 0.7) * 100 - 100);
  const peakCongestionZones = Math.round(congestionScore * 12 + activeSurges * 2);
  const affectedTransitRoutes = Math.round(congestionScore * 18 + activeSurges * 3);
  const estimatedPersonsAffected = Math.round(
    (congestionScore * 180000 + totalSurge * 40000) * timeMult,
  );

  const avgTravelTime = Math.round(35 * bprTravelTime(congestionScore, 0.7));
  const co2Increase = Math.round(avgDelayIncrease * 1.4 + activeSurges * 7);
  const emergencyDelayRisk =
    congestionScore > 0.80 ? ('critical' as const) :
    congestionScore > 0.60 ? ('high' as const) :
    congestionScore > 0.35 ? ('moderate' as const) :
    ('low' as const);

  return {
    congestionScore,
    avgDelayIncrease,
    peakCongestionZones,
    affectedTransitRoutes,
    estimatedPersonsAffected,
    avgTravelTime,
    co2Increase,
    emergencyDelayRisk,
  };
}

// Build GeoJSON FeatureCollection for neighborhood zone congestion overlay
export function generateZoneCongestionGeoJSON(
  venueSurges: Record<string, number>,
  globalIntensity: number,
  timeOfDay: number,
  customEvents: CustomTrafficEvent[] = [],
  staggeredArrivals = false,
): GeoJSON.FeatureCollection {
  const timeMult = getTimeMultiplier(timeOfDay);
  const staggerFactor = staggeredArrivals ? 0.75 : 1.0;

  const features: GeoJSON.Feature[] = LA_ZONES.map((zone) => {
    const [cLng, cLat] = zone.centroid;

    let surgePressure = 0;
    let minDistKm = Infinity;
    for (const [venueId, intensity] of Object.entries(venueSurges)) {
      if (intensity <= 0) continue;
      const venue = LA28_VENUES.find((v) => v.id === venueId);
      if (!venue) continue;
      const distKm = haversineDistance(cLng, cLat, venue.lng, venue.lat);
      minDistKm = Math.min(minDistKm, distKm);
      // Strong effect within 5 km, fades out past 18 km (tight zone of influence)
      const falloff = Math.max(0, 1 - distKm / 18);
      surgePressure = Math.max(surgePressure, intensity * falloff);
    }

    if (minDistKm === Infinity) {
      for (const venue of LA28_VENUES) {
        const distKm = haversineDistance(cLng, cLat, venue.lng, venue.lat);
        minDistKm = Math.min(minDistKm, distKm);
      }
    }

    let customPressure = 0;
    for (const event of customEvents) {
      const distKm = haversineDistance(cLng, cLat, event.lng, event.lat);
      const normalizedIntensity = Math.min(1, event.attendees / 70000);
      const decayKm = event.type === 'festival' ? 28 : event.type === 'rally' ? 32 : 22;
      const falloff = Math.max(0, 1 - distKm / decayKm);
      const timeSensitivity = getEventTimeSensitivity(event.type, timeOfDay);
      customPressure = Math.max(customPressure, normalizedIntensity * falloff * timeSensitivity);
      minDistKm = Math.min(minDistKm, distKm);
    }

    const base = zone.baseLoad * timeMult;
    // Proximity boost only within 10 km — keeps most of the city dark.
    // Base multiplier reduced to 0.9 (was 1.5) so low-intensity baseline stays calm.
    const proximityBoost = Math.max(0, 1 - minDistKm / 10) * globalIntensity * 0.35;
    const congestion = Math.min(
      1,
      (base * globalIntensity * 0.9 + proximityBoost + surgePressure * 0.85 + customPressure * 0.9) * staggerFactor,
    );

    return {
      type: 'Feature',
      geometry: { type: 'Polygon', coordinates: [zone.polygon] },
      properties: { id: zone.id, name: zone.name, congestion: +congestion.toFixed(3) },
    };
  });

  return { type: 'FeatureCollection', features };
}

export function generateTimelineData(
  globalIntensity: number,
  venueSurges: Record<string, number>,
) {
  const hours = ['6am', '7am', '8am', '9am', '10am', '12pm', '2pm', '4pm', '6pm', '8pm', '10pm'];
  const hourValues = [6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22];
  const totalSurge = Object.values(venueSurges).reduce((a, b) => a + b, 0);

  return hours.map((hour, i) => {
    const t = getTimeMultiplier(hourValues[i]);
    const baseline = t * 0.55;
    const sim = Math.min(1, t * globalIntensity + totalSurge * 0.12);
    return { hour, baseline: +baseline.toFixed(2), congestion: +sim.toFixed(2) };
  });
}

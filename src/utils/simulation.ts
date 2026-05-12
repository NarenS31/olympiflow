import { LA28_VENUES } from '../data/venues';
import { LA_ZONES } from '../data/zones';
import { haversineDistance } from './geoUtils';

// BPR (Bureau of Public Roads) travel-time function
// Returns travel time ratio: 1.0 = free flow, >1 = delayed
export function bprTravelTime(volume: number, capacity: number, alpha = 0.15, beta = 4): number {
  return 1 + alpha * Math.pow(volume / capacity, beta);
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

// Diffuse congestion from a surge epicenter in concentric rings
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
): GeoJSON.FeatureCollection {
  const timeMult = getTimeMultiplier(timeOfDay);
  const features: GeoJSON.Feature[] = [];

  // Scale base points by time and global intensity
  for (const p of basePoints) {
    features.push({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [p.lng, p.lat] },
      properties: { weight: Math.min(1, p.weight * timeMult * globalIntensity * 3) },
    });
  }

  // Add venue surge pressure rings
  for (const [venueId, intensity] of Object.entries(venueSurges)) {
    if (intensity <= 0) continue;
    const venue = LA28_VENUES.find((v) => v.id === venueId);
    if (!venue) continue;

    const surgePoints = createSurgePressurePoints(venue.lng, venue.lat, intensity);
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
): GeoJSON.FeatureCollection {
  const timeMult = getTimeMultiplier(timeOfDay);

  const features: GeoJSON.Feature[] = LA_ZONES.map((zone) => {
    const [cLng, cLat] = zone.centroid;

    // Pressure from each active venue surge, distance-weighted
    let surgePressure = 0;
    let minDistKm = Infinity;
    for (const [venueId, intensity] of Object.entries(venueSurges)) {
      if (intensity <= 0) continue;
      const venue = LA28_VENUES.find((v) => v.id === venueId);
      if (!venue) continue;
      const distKm = haversineDistance(cLng, cLat, venue.lng, venue.lat);
      minDistKm = Math.min(minDistKm, distKm);
      // Strong effect within 5 km, fades out past 30 km
      const falloff = Math.max(0, 1 - distKm / 30);
      surgePressure = Math.max(surgePressure, intensity * falloff);
    }

    // When no surges active, use nearest Olympic venue for proximity reference
    if (minDistKm === Infinity) {
      for (const venue of LA28_VENUES) {
        const distKm = haversineDistance(cLng, cLat, venue.lng, venue.lat);
        minDistKm = Math.min(minDistKm, distKm);
      }
    }

    const base = zone.baseLoad * timeMult;
    // Zones within ~20 km of the nearest active venue get a congestion boost that
    // scales with globalIntensity — so cranking the slider makes nearby zones go
    // redder faster while far zones stay comparatively green.
    const proximityBoost = Math.max(0, 1 - minDistKm / 20) * globalIntensity * 0.4;
    const congestion = Math.min(1, base * globalIntensity * 1.5 + proximityBoost + surgePressure * 0.85);

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

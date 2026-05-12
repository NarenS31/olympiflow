// Parse WKT MULTILINESTRING to GeoJSON
export function wktMultiLineStringToGeoJSON(wkt: string): GeoJSON.MultiLineString | null {
  try {
    const inner = wkt.replace(/^MULTILINESTRING\s*\(\(/, '').replace(/\)\)$/, '');
    const lineStrings = inner.split('),(');
    const coordinates = lineStrings.map((ls) =>
      ls.split(',').map((pair) => {
        const [lngStr, latStr] = pair.trim().split(/\s+/);
        return [parseFloat(lngStr), parseFloat(latStr)] as [number, number];
      }),
    );
    return { type: 'MultiLineString', coordinates };
  } catch {
    return null;
  }
}

// Parse "(lat, lng)" string to [lng, lat]
export function parseCoordinateString(s: string): [number, number] | null {
  const match = s.match(/\((-?[\d.]+),\s*(-?[\d.]+)\)/);
  if (!match) return null;
  const lat = parseFloat(match[1]);
  const lng = parseFloat(match[2]);
  if (isNaN(lat) || isNaN(lng)) return null;
  return [lng, lat];
}

export function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// Haversine distance in km
export function haversineDistance(
  lng1: number, lat1: number,
  lng2: number, lat2: number,
): number {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

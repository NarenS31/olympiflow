import { useEffect, useRef, useCallback } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useSimulationStore } from '../../stores/simulationStore';
import { LA28_VENUES } from '../../data/venues';
import { LA_ARTERIES, buildArteryGeoJSON } from '../../data/arteries';
import { generateHeatmapGeoJSON, generateZoneCongestionGeoJSON } from '../../utils/simulation';
import type { Venue, CustomTrafficEvent, TrafficEventType } from '../../types';

const LA_CENTER: [number, number] = [-118.2437, 34.0522];
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

// Animated dasharray sequence — "ant march" flowing transit lines
const DASH_SEQUENCE = [
  [0, 4, 3], [0.5, 4, 2.5], [1, 4, 2], [1.5, 4, 1.5],
  [2, 4, 1], [2.5, 4, 0.5], [3, 4, 0], [3.5, 3.5, 0],
];

// ── Particle system ──────────────────────────────────────────────────────────
interface Particle {
  coords: [number, number][];
  t: number;       // 0-1 progress along path
  speed: number;   // progress per frame
  size: number;    // dot radius px
  color: string;
  glowColor: string;
  opacity: number;
}

const P_COLORS = [
  { dot: '#cbd5e1', glow: '#94a3b8' },  // slate — most common
  { dot: '#cbd5e1', glow: '#94a3b8' },
  { dot: '#cbd5e1', glow: '#94a3b8' },
  { dot: '#93c5fd', glow: '#3b82f6' },  // blue
  { dot: '#a5b4fc', glow: '#6366f1' },  // indigo
  { dot: '#67e8f9', glow: '#0891b2' },  // teal
];

function initParticles(): Particle[] {
  const particles: Particle[] = [];
  for (const artery of LA_ARTERIES) {
    const count = 6 + Math.floor(Math.random() * 5); // 6-10 per artery
    for (let i = 0; i < count; i++) {
      const c = P_COLORS[Math.floor(Math.random() * P_COLORS.length)];
      particles.push({
        coords: artery.coords,
        t: Math.random(),
        speed: 0.00055 + Math.random() * 0.00085,
        size: 1.2 + Math.random() * 1.6,
        color: c.dot,
        glowColor: c.glow,
        opacity: 0.30 + Math.random() * 0.45,
      });
    }
  }
  return particles;
}

// ── City lights ───────────────────────────────────────────────────────────────
interface CityLight {
  lng: number;
  lat: number;
  size: number;
  color: string;
  glowColor: string;
  phase: number;    // sin wave offset (0–2π)
  speed: number;    // pulse speed
  baseOpacity: number;
}

// Palette: mostly warm sodium/amber streetlights, some cool LED/blue
const LIGHT_PALETTE = [
  { color: '#fffbeb', glow: '#fef3c7' },  // warm white
  { color: '#fffbeb', glow: '#fef3c7' },
  { color: '#fffbeb', glow: '#fef3c7' },
  { color: '#fde68a', glow: '#f59e0b' },  // amber
  { color: '#fde68a', glow: '#f59e0b' },
  { color: '#fed7aa', glow: '#fb923c' },  // orange
  { color: '#e0f9ff', glow: '#bae6fd' },  // cool white / LED
];

// Rough LA coastline waypoints [lat, coastLng] south→north.
// Anything west of coastLng at a given lat is the Pacific Ocean.
const COASTLINE: [number, number][] = [
  [33.70, -118.22],
  [33.74, -118.32],
  [33.76, -118.41],
  [33.80, -118.40],
  [33.85, -118.39],
  [33.90, -118.40],
  [33.93, -118.42],
  [33.97, -118.45],
  [33.99, -118.47],
  [34.01, -118.50],
  [34.04, -118.54],
  [34.08, -118.58],
  [34.12, -118.62],
  [34.22, -118.70],
];

function isOnLand(lng: number, lat: number): boolean {
  for (let i = 0; i < COASTLINE.length - 1; i++) {
    const [lat1, lng1] = COASTLINE[i];
    const [lat2, lng2] = COASTLINE[i + 1];
    if (lat >= lat1 && lat < lat2) {
      const t = (lat - lat1) / (lat2 - lat1);
      const coastLng = lng1 + t * (lng2 - lng1);
      return lng > coastLng; // east of coast = land
    }
  }
  return true;
}

function initCityLights(): CityLight[] {
  const lights: CityLight[] = [];
  // Grid over the LA basin with jitter so it looks organic, not perfectly aligned
  const lngMin = -118.555, lngMax = -118.095;
  const latMin = 33.745,  latMax = 34.220;
  const step = 0.0055; // ~600m grid spacing
  for (let lng = lngMin; lng <= lngMax; lng += step) {
    for (let lat = latMin; lat <= latMax; lat += step) {
      if (Math.random() > 0.38) continue; // sparse — not every block visible
      const jLng = lng + (Math.random() - 0.5) * step * 0.9;
      const jLat = lat + (Math.random() - 0.5) * step * 0.9;
      if (!isOnLand(jLng, jLat)) continue; // skip ocean
      const p = LIGHT_PALETTE[Math.floor(Math.random() * LIGHT_PALETTE.length)];
      lights.push({
        lng: jLng,
        lat: jLat,
        size: 0.55 + Math.random() * 1.05,
        color: p.color,
        glowColor: p.glow,
        phase: Math.random() * Math.PI * 2,
        speed: 0.00022 + Math.random() * 0.00055,
        baseOpacity: 0.10 + Math.random() * 0.26,
      });
    }
  }
  return lights;
}

// ── Venue risk data ──────────────────────────────────────────────────────────
const VENUE_RISK_DATA: Record<string, { abbr: string; risk: number; athletes?: number; dashRoutes: number }> = {
  'la-coliseum':      { abbr: 'COL',  risk: 78.1, athletes: 2269, dashRoutes: 1 },
  'long-beach-arena': { abbr: 'LB',   risk: 35.3, athletes: 546,  dashRoutes: 0 },
  'crypto-arena':     { abbr: 'CC',   risk: 35.0,                  dashRoutes: 5 },
  'sofi':             { abbr: 'SOFI', risk: 33.9,                  dashRoutes: 0 },
  'rose-bowl':        { abbr: 'RB',   risk: 33.9,                  dashRoutes: 0 },
  'pauley':           { abbr: 'PAU',  risk: 20.0,                  dashRoutes: 2 },
  'intuit-dome':      { abbr: 'INT',  risk: 28.5,                  dashRoutes: 0 },
  'bmo-stadium':      { abbr: 'BMO',  risk: 25.0,                  dashRoutes: 1 },
  'sepulveda-basin':  { abbr: 'SEP',  risk: 18.0,                  dashRoutes: 0 },
  'el-dorado':        { abbr: 'ELD',  risk: 15.0,                  dashRoutes: 0 },
  'dignity-health':   { abbr: 'DHS',  risk: 22.0,                  dashRoutes: 0 },
  'ucla-olympic':     { abbr: 'UCLA', risk: 20.0,                  dashRoutes: 3 },
};

function riskColor(risk: number): string {
  if (risk > 60) return '#FF453A';
  if (risk > 30) return '#FF9F0A';
  return '#30D158';
}

function riskBadgeLabel(risk: number): string {
  if (risk > 60) return 'HIGH';
  if (risk > 30) return 'MODERATE';
  return 'LOW';
}

// ── Hexagonal GIS-style venue markers ────────────────────────────────────────
const HEX_CLIP = 'polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%)';

function injectHexStyles() {
  if (document.getElementById('hex-marker-style')) return;
  const s = document.createElement('style');
  s.id = 'hex-marker-style';
  s.textContent = `
    .hex-marker { transition: transform 0.18s ease; transform-origin: center; cursor: pointer; }
    .hex-marker:hover { transform: scale(1.18); }
    .col-ring { position:absolute; border-radius:50%; pointer-events:none; border-style:solid; }
    .col-ring-1 { animation: coliseum-ring-pulse 4s ease-out infinite; }
    .col-ring-2 { animation: coliseum-ring-pulse 4s ease-out 1.33s infinite; }
    .col-ring-3 { animation: coliseum-ring-pulse 4s ease-out 2.66s infinite; }
    @keyframes coliseum-ring-pulse {
      0%   { transform: translate(-50%,-50%) scale(0.85); opacity: 0.20; }
      60%  { transform: translate(-50%,-50%) scale(1.20); opacity: 0.08; }
      100% { transform: translate(-50%,-50%) scale(1.40); opacity: 0; }
    }
  `;
  document.head.appendChild(s);
}

const VENUE_MARKER_COLORS: Record<string, { fill: string; border: string; text: string; riskText: string }> = {
  'la-coliseum':      { fill: '#7B1C1C', border: '#FF453A', text: '#FFFFFF', riskText: '#FF8A84' },
  'long-beach-arena': { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' },
  'crypto-arena':     { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' },
  'sofi':             { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' },
  'rose-bowl':        { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' },
  'intuit-dome':      { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' },
  'pauley':           { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' },
  'bmo-stadium':      { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' },
  'sepulveda-basin':  { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' },
  'el-dorado':        { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' },
  'dignity-health':   { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' },
  'ucla-olympic':     { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' },
};

function createVenueMarkerElement(venue: Venue): HTMLElement {
  injectHexStyles();

  const rd         = VENUE_RISK_DATA[venue.id];
  const abbr       = rd?.abbr ?? venue.shortName.substring(0, 4).toUpperCase();
  const colors     = VENUE_MARKER_COLORS[venue.id] ?? { fill: '#1E3A5F', border: '#0A84FF', text: '#FFFFFF', riskText: '#5B9BD5' };
  const isHighRisk = venue.id === 'la-coliseum';

  const size     = isHighRisk ? 46 : 34;
  const fontSize = isHighRisk ? '9px' : '7.5px';

  const wrapper = document.createElement('div');
  wrapper.style.cssText = `position:relative;width:${size}px;height:${size}px;cursor:pointer;will-change:transform;`;

  if (isHighRisk) {
    const ringSize = size + 18;
    for (let i = 1; i <= 3; i++) {
      const ring = document.createElement('div');
      ring.className = `col-ring col-ring-${i}`;
      ring.style.cssText = `
        width:${ringSize}px;height:${ringSize}px;
        top:50%;left:50%;
        border:2px solid #FF453A;
        pointer-events:none;
        opacity:0.2;
      `;
      wrapper.appendChild(ring);
    }
  }

  const hex = document.createElement('div');
  hex.className = 'hex-marker';
  hex.style.cssText = `
    position:absolute;inset:0;
    clip-path:${HEX_CLIP};
    background:${colors.fill};
    box-shadow:inset 0 0 0 ${isHighRisk ? '2px' : '1.5px'} ${colors.border};
    display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1px;
    z-index:1;
  `;

  const label = document.createElement('span');
  label.style.cssText = `font-family:'SF Mono',ui-monospace,monospace;font-size:${fontSize};font-weight:700;color:${colors.text};letter-spacing:0.04em;text-transform:uppercase;pointer-events:none;line-height:1;text-align:center;`;
  label.textContent = abbr;
  hex.appendChild(label);

  if (rd?.risk) {
    const riskLabel = document.createElement('span');
    riskLabel.style.cssText = `font-family:'SF Mono',ui-monospace,monospace;font-size:7px;font-weight:500;color:${colors.riskText};pointer-events:none;line-height:1;text-align:center;`;
    riskLabel.textContent = rd.risk.toFixed(1);
    hex.appendChild(riskLabel);
  }

  wrapper.appendChild(hex);
  return wrapper;
}

function createVenuePopupHTML(venue: Venue): string {
  const rd    = VENUE_RISK_DATA[venue.id];
  const risk  = rd?.risk;
  const rc    = risk ? riskColor(risk) : '#8B949E';
  const badge = risk ? riskBadgeLabel(risk) : '';

  const riskRow = risk ? `
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
      <span style="font-size:10px;color:rgba(255,255,255,0.35);letter-spacing:0.06em;text-transform:uppercase;font-weight:500;">Risk</span>
      <span style="font-size:22px;font-weight:200;color:${rc};font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display',sans-serif;letter-spacing:-0.02em;">${risk.toFixed(1)}</span>
      <span style="font-size:10px;padding:2px 8px;border-radius:20px;color:${rc};background:${rc}18;font-weight:600;letter-spacing:0.04em;">${badge}</span>
    </div>
  ` : '';

  const athleteRow = rd?.athletes ? `
    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:0.5px solid rgba(255,255,255,0.06);">
      <span style="font-size:10px;color:rgba(255,255,255,0.35);font-weight:500;letter-spacing:0.04em;text-transform:uppercase;">Athletes</span>
      <span style="font-size:11px;color:rgba(255,255,255,0.7);font-family:'SF Mono',ui-monospace,monospace;">${rd.athletes.toLocaleString()}</span>
    </div>
  ` : '';

  const dashRow = rd !== undefined ? `
    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:0.5px solid rgba(255,255,255,0.06);">
      <span style="font-size:10px;color:rgba(255,255,255,0.35);font-weight:500;letter-spacing:0.04em;text-transform:uppercase;">DASH Routes</span>
      <span style="font-size:11px;font-family:'SF Mono',ui-monospace,monospace;font-weight:600;color:${rd.dashRoutes > 0 ? '#0A84FF' : '#FF453A'};">
        ${rd.dashRoutes > 0 ? `${rd.dashRoutes} route${rd.dashRoutes !== 1 ? 's' : ''}` : 'None'}
      </span>
    </div>
  ` : '';

  const sports = venue.sports
    .map((s) => `<span style="display:inline-block;margin:2px 3px 2px 0;padding:2px 8px;border-radius:6px;background:rgba(255,255,255,0.06);font-size:10px;color:rgba(255,255,255,0.45);font-weight:400;letter-spacing:0.03em;">${s}</span>`)
    .join('');

  return `
    <div style="font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text','Helvetica Neue',sans-serif;color:rgba(255,255,255,0.85);padding:14px 16px;min-width:220px;">
      <div style="font-size:10px;color:rgba(255,255,255,0.35);font-weight:500;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:3px;">
        ${venue.neighborhood}&nbsp;·&nbsp;${venue.capacity.toLocaleString()} seats
      </div>
      <div style="font-size:15px;font-weight:500;color:rgba(255,255,255,0.9);margin-bottom:10px;line-height:1.2;letter-spacing:-0.01em;">${venue.name}</div>
      ${riskRow}
      <div style="border-top:0.5px solid rgba(255,255,255,0.08);padding-top:8px;margin-bottom:8px;">
        ${athleteRow}
        ${dashRow}
        <div style="display:flex;justify-content:space-between;padding-top:4px;">
          <span style="font-size:10px;color:rgba(255,255,255,0.35);font-weight:500;letter-spacing:0.04em;text-transform:uppercase;">Capacity</span>
          <span style="font-size:11px;color:rgba(255,255,255,0.7);font-family:'SF Mono',ui-monospace,monospace;">${venue.capacity.toLocaleString()}</span>
        </div>
      </div>
      <div>${sports}</div>
    </div>
  `;
}

// ── Staggered arrivals overlay (concentric rings around Coliseum) ─────────────
const COLISEUM_CENTER: [number, number] = [-118.2879, 34.0140];

function makeCirclePolygon(center: [number, number], radiusKm: number, steps = 40): GeoJSON.Feature {
  const [lng, lat] = center;
  const lngPerKm   = 1 / (111.32 * Math.cos((lat * Math.PI) / 180));
  const latPerKm   = 1 / 110.574;
  const coords: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const a = (i / steps) * Math.PI * 2;
    coords.push([lng + Math.cos(a) * radiusKm * lngPerKm, lat + Math.sin(a) * radiusKm * latPerKm]);
  }
  return { type: 'Feature', geometry: { type: 'Polygon', coordinates: [coords] }, properties: { r: radiusKm } };
}

const STAGGER_ZONES: GeoJSON.FeatureCollection = {
  type: 'FeatureCollection',
  features: [0.6, 1.1, 1.7, 2.3].map((r) => makeCirclePolygon(COLISEUM_CENTER, r)),
};

// ── Custom event markers ─────────────────────────────────────────────────────
const CUSTOM_EVENT_COLORS: Record<TrafficEventType, string> = {
  sports:  '#0891b2',
  concert: '#7c3aed',
  festival:'#16a34a',
  rally:   '#d97706',
};

function createCustomEventMarker(event: CustomTrafficEvent): HTMLElement {
  const color = CUSTOM_EVENT_COLORS[event.type];
  const size  = Math.round(32 + (event.attendees / 100000) * 16); // 32–48 px

  if (!document.getElementById('custom-evt-style')) {
    const s = document.createElement('style');
    s.id = 'custom-evt-style';
    s.textContent = `
      @keyframes evt-pulse { 0%,100%{transform:scale(1);opacity:.5} 50%{transform:scale(1.35);opacity:.18} }
      .evt-ring { animation: evt-pulse 2.2s ease-in-out infinite; }
    `;
    document.head.appendChild(s);
  }

  const wrap = document.createElement('div');
  wrap.style.cssText = `position:relative;width:${size}px;height:${size}px;display:flex;align-items:center;justify-content:center;cursor:pointer;`;

  const ring = document.createElement('div');
  ring.className = 'evt-ring';
  ring.style.cssText = `position:absolute;inset:-5px;border-radius:50%;border:2px solid ${color};pointer-events:none;`;

  const core = document.createElement('div');
  core.style.cssText = `
    width:${size}px;height:${size}px;border-radius:50%;
    background:rgba(4,8,15,0.92);border:2px solid ${color};
    display:flex;align-items:center;justify-content:center;
    color:${color};font-size:${size < 38 ? 9 : 10}px;font-weight:700;font-family:monospace;
    box-shadow:0 0 14px ${color}55;backdrop-filter:blur(8px);
    transition:transform .18s ease,box-shadow .18s ease;
  `;
  const label = event.attendees >= 1000 ? `${Math.round(event.attendees / 1000)}k` : String(event.attendees);
  core.textContent = label;

  wrap.appendChild(ring);
  wrap.appendChild(core);
  wrap.addEventListener('mouseenter', () => { core.style.transform = 'scale(1.15)'; core.style.boxShadow = `0 0 22px ${color}88`; });
  wrap.addEventListener('mouseleave', () => { core.style.transform = 'scale(1)';    core.style.boxShadow = `0 0 14px ${color}55`; });

  return wrap;
}

// ── Component ────────────────────────────────────────────────────────────────
export function OlympiMap() {
  const mapContainer          = useRef<HTMLDivElement>(null);
  const canvasRef             = useRef<HTMLCanvasElement>(null);
  const map                   = useRef<maplibregl.Map | null>(null);
  const markersRef            = useRef<maplibregl.Marker[]>([]);
  const customEventMarkersRef = useRef<maplibregl.Marker[]>([]);
  const isLoaded              = useRef(false);
  const animRafRef            = useRef<number>(0);
  const particleRafRef        = useRef<number>(0);
  const particlesRef          = useRef<Particle[]>([]);
  const cityLightsRef         = useRef<CityLight[]>([]);

  const venueSurges           = useSimulationStore((s) => s.venueSurges);
  const globalIntensity       = useSimulationStore((s) => s.globalIntensity);
  const timeOfDay             = useSimulationStore((s) => s.timeOfDay);
  const layers                = useSimulationStore((s) => s.layers);
  const staggeredArrivals     = useSimulationStore((s) => s.staggeredArrivals);
  const transitData           = useSimulationStore((s) => s.transitData);
  const heatmapBaseData       = useSimulationStore((s) => s.heatmapBaseData);
  const crimeData             = useSimulationStore((s) => s.crimeData);
  const selectVenue           = useSimulationStore((s) => s.selectVenue);
  const customEvents          = useSimulationStore((s) => s.customEvents);
  const placingEvent          = useSimulationStore((s) => s.placingEvent);
  const setPendingEventLocation = useSimulationStore((s) => s.setPendingEventLocation);
  const setPlacingEvent      = useSimulationStore((s) => s.setPlacingEvent);

  const getBasePoints = useCallback(() => {
    if (!heatmapBaseData) return [];
    return heatmapBaseData.features.map((f) => {
      const g = f.geometry as GeoJSON.Point;
      return { lng: g.coordinates[0], lat: g.coordinates[1], weight: (f.properties?.weight as number) ?? 0.5 };
    });
  }, [heatmapBaseData]);

  // ── Map + particle init ──────────────────────────────────────────────────
  useEffect(() => {
    if (map.current || !mapContainer.current) return;

    // Size canvas to fill the map container (set after first paint)
    const sizeCanvas = () => {
      const canvas = canvasRef.current;
      const container = mapContainer.current;
      if (!canvas || !container) return;
      canvas.width  = container.offsetWidth;
      canvas.height = container.offsetHeight;
    };
    // Defer until after layout so offsetWidth/Height are non-zero
    requestAnimationFrame(sizeCanvas);
    window.addEventListener('resize', sizeCanvas);

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: MAP_STYLE,
      center: LA_CENTER,
      zoom: 10,
      minZoom: 8,
      maxZoom: 17,
      pitch: 0,
    });

    map.current.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    map.current.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: 'metric' }), 'bottom-left');

    map.current.on('load', () => {
      const m = map.current!;
      isLoaded.current = true;

      // ── Artery road glow ─────────────────────────────────────────────────
      m.addSource('arteries-source', {
        type: 'geojson',
        data: buildArteryGeoJSON() as GeoJSON.FeatureCollection,
      });
      m.addLayer({ id: 'artery-bloom', type: 'line', source: 'arteries-source',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#9a3412', 'line-width': 18, 'line-opacity': 0, 'line-blur': 6 },
      });
      m.addLayer({ id: 'artery-glow', type: 'line', source: 'arteries-source',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#7f1d1d', 'line-width': 8, 'line-opacity': 0, 'line-blur': 3 },
      });
      m.addLayer({ id: 'artery-core', type: 'line', source: 'arteries-source',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#9a3412', 'line-width': 2, 'line-opacity': 0 },
      });

      // ── Heatmap — added BEFORE zones so zones render on top ─────────────
      m.addSource('heatmap-source', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      m.addLayer({
        id: 'traffic-heatmap', type: 'heatmap', source: 'heatmap-source',
        paint: {
          'heatmap-weight': ['interpolate', ['linear'], ['get', 'weight'], 0, 0, 1, 1],
          // Low intensity — venue hotspots already have high weights so multiplying less is fine
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 8, 0.28, 11, 0.55, 14, 0.95],
          // Amber → orange → red — only reaches red at high density (>0.80)
          'heatmap-color': ['interpolate', ['linear'], ['heatmap-density'],
            0,    'rgba(0,0,0,0)',
            0.10, 'rgba(0,0,0,0)',
            0.25, 'rgba(245,158,11,0.28)',   // amber starts
            0.44, 'rgba(222,100,15,0.52)',   // amber-orange
            0.62, 'rgba(210,55,10,0.70)',    // deep orange
            0.80, 'rgba(220,38,38,0.84)',    // #DC2626 red
            1.0,  'rgba(200,18,18,0.92)',    // deep red
          ],
          // Smaller radius so venue hotspots stay tight and don't bleed into each other
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 8, 11, 11, 22, 14, 42],
          'heatmap-opacity': 0.85,
        },
      });

      // ── Crime heatmap ─────────────────────────────────────────────────────
      m.addSource('crime-source', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      m.addLayer({
        id: 'crime-heatmap', type: 'heatmap', source: 'crime-source',
        layout: { visibility: 'none' },
        paint: {
          'heatmap-weight': ['interpolate', ['linear'], ['get', 'weight'], 0, 0, 1, 1],
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 8, 0.7, 14, 2.0],
          'heatmap-color': ['interpolate', ['linear'], ['heatmap-density'],
            0,    'rgba(0,0,0,0)',
            0.15, 'rgba(88,0,120,0.22)',
            0.35, 'rgba(130,0,160,0.44)',
            0.60, 'rgba(180,0,140,0.62)',
            0.80, 'rgba(210,20,80,0.76)',
            1.0,  'rgba(230,50,30,0.88)',
          ],
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 8, 14, 14, 30],
          'heatmap-opacity': 0.60,
        },
      });

      // ── Zone overlay — on top of heatmap ─────────────────────────────────
      m.addSource('zones-source', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      // Outer halo: wide blurred line that feathers zone edges so they look organic, not boxy
      m.addLayer({
        id: 'zone-halo', type: 'line', source: 'zones-source',
        paint: {
          'line-color': ['interpolate', ['linear'], ['get', 'congestion'],
            0, '#14532d', 0.35, '#92400e', 0.65, '#991b1b', 1.0, '#7f1d1d'],
          'line-width': 22,
          'line-blur': 16,
          'line-opacity': ['interpolate', ['linear'], ['get', 'congestion'],
            0, 0.0, 0.30, 0.0, 0.50, 0.16, 1.0, 0.36],
        },
      });
      m.addLayer({
        id: 'zone-fill', type: 'fill', source: 'zones-source',
        paint: {
          'fill-color': ['interpolate', ['linear'], ['get', 'congestion'],
            0, '#166534', 0.28, '#854d0e', 0.50, '#9a3412', 0.72, '#991b1b', 1.0, '#7f1d1d'],
          'fill-opacity': ['interpolate', ['linear'], ['get', 'congestion'],
            0, 0.0, 0.25, 0.0, 0.40, 0.06, 0.58, 0.15, 0.78, 0.26, 1.0, 0.36],
          'fill-antialias': true,
        },
      });
      // Crisp inner border for definition
      m.addLayer({
        id: 'zone-border', type: 'line', source: 'zones-source',
        paint: {
          'line-color': ['interpolate', ['linear'], ['get', 'congestion'],
            0, '#166534', 0.5, '#9a3412', 1.0, '#991b1b'],
          'line-width': 1,
          'line-opacity': ['interpolate', ['linear'], ['get', 'congestion'],
            0, 0.0, 0.15, 0.15, 0.50, 0.40, 1.0, 0.65],
        },
      });

      // ── Transit routes + animated flow ───────────────────────────────────
      m.addSource('transit-source', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      m.addLayer({ id: 'transit-casing', type: 'line', source: 'transit-source',
        layout: { 'line-cap': 'round' },
        paint: { 'line-color': '#000', 'line-width': 6, 'line-opacity': 0.4 },
      });
      m.addLayer({ id: 'transit-routes', type: 'line', source: 'transit-source',
        layout: { 'line-cap': 'round' },
        paint: { 'line-color': ['coalesce', ['get', 'color'], '#6d28d9'], 'line-width': 2.5, 'line-opacity': 0.55 },
      });
      m.addLayer({ id: 'transit-flow', type: 'line', source: 'transit-source',
        layout: { 'line-cap': 'round' },
        paint: { 'line-color': ['coalesce', ['get', 'color'], '#7c3aed'], 'line-width': 2, 'line-dasharray': [0, 4, 3], 'line-opacity': 0.85 },
      });

      // ── Dash animation loop ───────────────────────────────────────────────
      let lastDashStep = -1, lastDashTs = 0;
      function animateDash(ts: number) {
        if (ts - lastDashTs >= 80) {
          const step = Math.floor(ts / 80) % DASH_SEQUENCE.length;
          if (step !== lastDashStep && m.getLayer('transit-flow')) {
            m.setPaintProperty('transit-flow', 'line-dasharray', DASH_SEQUENCE[step]);
            lastDashStep = step;
          }
          lastDashTs = ts;
        }
        animRafRef.current = requestAnimationFrame(animateDash);
      }
      animRafRef.current = requestAnimationFrame(animateDash);

      // ── Particle animation loop ───────────────────────────────────────────
      particlesRef.current = initParticles();
      cityLightsRef.current = initCityLights();

      function animateParticles() {
        const canvas = canvasRef.current;
        if (!canvas || !m) { particleRafRef.current = requestAnimationFrame(animateParticles); return; }

        const ctx = canvas.getContext('2d')!;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const intensity = useSimulationStore.getState().globalIntensity;
        const speedMult = 0.5 + intensity * 1.8;

        // ── City lights — drawn first so particles appear above ─────────────
        const now = Date.now();
        const zoom = m.getZoom();
        // Fade lights in between zoom 8.5-10.5 so they don't overwhelm at wide view
        const lightAlphaScale = Math.max(0, Math.min(1, (zoom - 8.5) / 2.0));
        if (lightAlphaScale > 0) {
          // Two-pass: tiny dots first (no shadow, fast), then glow on larger lights
          ctx.save();
          ctx.shadowBlur = 0;
          for (const light of cityLightsRef.current) {
            const { x, y } = m.project([light.lng, light.lat] as [number, number]);
            if (x < -8 || x > canvas.width + 8 || y < -8 || y > canvas.height + 8) continue;
            const pulse = 0.52 + 0.48 * Math.sin(now * light.speed + light.phase);
            ctx.globalAlpha = light.baseOpacity * pulse * lightAlphaScale;
            ctx.fillStyle = light.color;
            ctx.beginPath();
            ctx.arc(x, y, light.size, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.restore();
          // Glow pass — only larger lights get the expensive shadowBlur
          for (const light of cityLightsRef.current) {
            if (light.size < 1.0) continue;
            const { x, y } = m.project([light.lng, light.lat] as [number, number]);
            if (x < -8 || x > canvas.width + 8 || y < -8 || y > canvas.height + 8) continue;
            const pulse = 0.52 + 0.48 * Math.sin(now * light.speed + light.phase);
            ctx.save();
            ctx.globalAlpha = light.baseOpacity * pulse * lightAlphaScale * 0.7;
            ctx.shadowBlur = light.size * 7;
            ctx.shadowColor = light.glowColor;
            ctx.fillStyle = light.color;
            ctx.beginPath();
            ctx.arc(x, y, light.size, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
          }
        }

        for (const p of particlesRef.current) {
          p.t += p.speed * speedMult;
          if (p.t > 1) p.t -= 1;

          const segCount = p.coords.length - 1;
          const rawSeg   = p.t * segCount;
          const segIdx   = Math.min(Math.floor(rawSeg), segCount - 1);
          const segT     = rawSeg - segIdx;

          const [lng1, lat1] = p.coords[segIdx];
          const [lng2, lat2] = p.coords[segIdx + 1];
          const lng = lng1 + (lng2 - lng1) * segT;
          const lat = lat1 + (lat2 - lat1) * segT;

          const { x, y } = m.project([lng, lat] as [number, number]);
          if (x < -30 || x > canvas.width + 30 || y < -30 || y > canvas.height + 30) continue;

          // Shift toward orange/red at high intensity
          let color    = p.color;
          let glow     = p.glowColor;
          if (intensity > 0.75) {
            color = '#fca5a5'; glow = '#dc2626';
          } else if (intensity > 0.50) {
            color = '#fed7aa'; glow = '#d97706';
          }

          const alpha = p.opacity * (0.65 + intensity * 0.35);
          ctx.save();
          ctx.globalAlpha = alpha;
          ctx.shadowBlur  = p.size * 6;
          ctx.shadowColor = glow;
          ctx.beginPath();
          ctx.arc(x, y, p.size, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.restore();
        }

        particleRafRef.current = requestAnimationFrame(animateParticles);
      }
      particleRafRef.current = requestAnimationFrame(animateParticles);

      // ── Staggered arrival zones (Coliseum concentric rings) ──────────────────
      m.addSource('stagger-zones', { type: 'geojson', data: STAGGER_ZONES });
      m.addLayer({
        id: 'stagger-fill', type: 'fill', source: 'stagger-zones',
        layout: { visibility: 'none' },
        paint: { 'fill-color': '#16A34A', 'fill-opacity': 0.06 },
      });
      m.addLayer({
        id: 'stagger-border', type: 'line', source: 'stagger-zones',
        layout: { visibility: 'none' },
        paint: { 'line-color': '#16A34A', 'line-width': 1.5, 'line-opacity': 0.6, 'line-dasharray': [4, 2] },
      });

      // ── Coordinate display ────────────────────────────────────────────────────
      m.on('mousemove', (e) => {
        const el = document.getElementById('coord-display');
        if (el) {
          const lat = e.lngLat.lat.toFixed(4);
          const lng = Math.abs(e.lngLat.lng).toFixed(4);
          const ns  = e.lngLat.lat >= 0 ? 'N' : 'S';
          const ew  = e.lngLat.lng <  0 ? 'W' : 'E';
          el.textContent = `${lat}° ${ns}, ${lng}° ${ew}`;
        }
      });

      // ── Venue markers ─────────────────────────────────────────────────────
      for (const venue of LA28_VENUES) {
        const el = createVenueMarkerElement(venue);
        const popup = new maplibregl.Popup({ offset: [0, -30], closeButton: false, className: 'olympi-popup', maxWidth: '270px' })
          .setHTML(createVenuePopupHTML(venue));
        const marker = new maplibregl.Marker({ element: el, anchor: 'center', pitchAlignment: 'map', rotationAlignment: 'map' })
          .setLngLat([venue.lng, venue.lat]).setPopup(popup).addTo(m);
        el.addEventListener('click', () => selectVenue(venue.id));
        markersRef.current.push(marker);
      }

      // ── Global styles ─────────────────────────────────────────────────────
      // Inject SVG noise filter for glass texture
      if (!document.getElementById('glass-noise-svg')) {
        const noiseSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        noiseSvg.id = 'glass-noise-svg';
        noiseSvg.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden;';
        noiseSvg.innerHTML = `<defs>
          <filter id="glass-noise" x="0%" y="0%" width="100%" height="100%">
            <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch"/>
            <feColorMatrix type="saturate" values="0"/>
            <feBlend in="SourceGraphic" mode="overlay" result="blend"/>
            <feComposite in="blend" in2="SourceGraphic" operator="in"/>
          </filter>
        </defs>`;
        document.body.appendChild(noiseSvg);
      }

      const style = document.createElement('style');
      style.textContent = `
        .olympi-popup .maplibregl-popup-content {
          background:rgba(17,19,24,0.97);border:0.5px solid rgba(255,255,255,0.1);border-radius:12px;
          padding:0;box-shadow:0 12px 48px rgba(0,0,0,0.8);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
        }
        .olympi-popup .maplibregl-popup-tip { display:none; }
        .maplibregl-ctrl-attrib { display:none !important; }
        .maplibregl-ctrl-group {
          background:rgba(28,28,30,0.88)!important;border:0.5px solid rgba(255,255,255,0.1)!important;
          border-radius:10px!important;overflow:hidden;backdrop-filter:blur(20px)!important;box-shadow:0 4px 20px rgba(0,0,0,0.4);
        }
        .maplibregl-ctrl-group button {
          background:transparent!important;color:rgba(255,255,255,0.35)!important;border:none!important;width:34px!important;height:34px!important;
        }
        .maplibregl-ctrl-group button:hover{background:rgba(255,255,255,0.06)!important;color:rgba(255,255,255,0.8)!important;}
        .maplibregl-ctrl-group button+button{border-top:0.5px solid rgba(255,255,255,0.08)!important;}
      `;
      document.head.appendChild(style);
    });

    return () => {
      window.removeEventListener('resize', sizeCanvas);
      cancelAnimationFrame(animRafRef.current);
      cancelAnimationFrame(particleRafRef.current);
      map.current?.remove();
      map.current = null;
      isLoaded.current = false;
      markersRef.current = [];
    };
  }, [selectVenue]);

  // ── Reactive updates ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    (map.current.getSource('zones-source') as maplibregl.GeoJSONSource)
      ?.setData(generateZoneCongestionGeoJSON(venueSurges, globalIntensity, timeOfDay, customEvents, staggeredArrivals));
  }, [venueSurges, globalIntensity, timeOfDay, customEvents, staggeredArrivals]);

  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    const pts = getBasePoints();
    const geojson = generateHeatmapGeoJSON(pts, venueSurges, globalIntensity, timeOfDay, customEvents, staggeredArrivals);
    (map.current.getSource('heatmap-source') as maplibregl.GeoJSONSource)?.setData(geojson);
  }, [venueSurges, globalIntensity, timeOfDay, heatmapBaseData, getBasePoints, customEvents, staggeredArrivals]);

  // ── Click-to-place mode ───────────────────────────────────────────────────
  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    const m = map.current;
    if (!placingEvent) {
      m.getCanvas().style.cursor = '';
      return;
    }
    m.getCanvas().style.cursor = 'crosshair';
    const handleClick = (e: maplibregl.MapMouseEvent) => {
      setPendingEventLocation({ lng: e.lngLat.lng, lat: e.lngLat.lat });
      setPlacingEvent(false);
    };
    m.once('click', handleClick);
    return () => { m.off('click', handleClick); m.getCanvas().style.cursor = ''; };
  }, [placingEvent, setPendingEventLocation, setPlacingEvent]);

  // ── Custom event markers ──────────────────────────────────────────────────
  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    customEventMarkersRef.current.forEach((mk) => mk.remove());
    customEventMarkersRef.current = [];
    for (const event of customEvents) {
      const el = createCustomEventMarker(event);
      const marker = new maplibregl.Marker({ element: el, anchor: 'center', pitchAlignment: 'map', rotationAlignment: 'map' })
        .setLngLat([event.lng, event.lat])
        .addTo(map.current!);
      customEventMarkersRef.current.push(marker);
    }
  }, [customEvents]);

  useEffect(() => {
    if (!isLoaded.current || !map.current || !transitData) return;
    (map.current.getSource('transit-source') as maplibregl.GeoJSONSource)?.setData(transitData);
  }, [transitData]);

  useEffect(() => {
    if (!isLoaded.current || !map.current || !crimeData) return;
    (map.current.getSource('crime-source') as maplibregl.GeoJSONSource)?.setData(crimeData);
  }, [crimeData]);

  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    const m = map.current;
    if (!m.getLayer('artery-bloom')) return;
    const bloomOp = Math.max(0, (globalIntensity - 0.2) * 0.22);
    const glowOp  = Math.max(0, (globalIntensity - 0.2) * 0.38);
    const coreOp  = Math.max(0, (globalIntensity - 0.15) * 0.65);
    m.setPaintProperty('artery-bloom', 'line-opacity', bloomOp);
    m.setPaintProperty('artery-glow',  'line-opacity', glowOp);
    m.setPaintProperty('artery-core',  'line-opacity', coreOp);
    const color = globalIntensity > 0.7 ? '#991b1b' : globalIntensity > 0.45 ? '#9a3412' : '#854d0e';
    m.setPaintProperty('artery-bloom', 'line-color', color);
    m.setPaintProperty('artery-glow',  'line-color', color);
    m.setPaintProperty('artery-core',  'line-color', color);
  }, [globalIntensity]);

  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    const m = map.current;
    const vis = (id: string, on: boolean) => {
      if (m.getLayer(id)) m.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none');
    };
    vis('zone-halo', layers.heatmap); vis('zone-fill', layers.heatmap); vis('zone-border', layers.heatmap);
    vis('traffic-heatmap', layers.heatmap);
    vis('artery-bloom', layers.heatmap); vis('artery-glow', layers.heatmap); vis('artery-core', layers.heatmap);
    vis('transit-routes', layers.transit); vis('transit-casing', layers.transit); vis('transit-flow', layers.transit);
    vis('crime-heatmap', layers.crime);
    markersRef.current.forEach((mk) => { mk.getElement().style.display = layers.venues ? 'flex' : 'none'; });
  }, [layers]);

  // Staggered arrivals overlay
  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    const m   = map.current;
    const vis = staggeredArrivals ? 'visible' : 'none';
    if (m.getLayer('stagger-fill'))   m.setLayoutProperty('stagger-fill',   'visibility', vis);
    if (m.getLayer('stagger-border')) m.setLayoutProperty('stagger-border', 'visibility', vis);
  }, [staggeredArrivals]);

  return (
    <div
      ref={mapContainer}
      className="w-full h-full"
      style={{ background: '#04080f', position: 'relative' }}
    >
      {/* Particle canvas sits above MapLibre's canvas; pointer-events disabled so map stays interactive */}
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          pointerEvents: 'none',
          zIndex: 2,
          mixBlendMode: 'screen',
        }}
      />
      {/* Coordinate display — updates on mousemove via DOM id */}
      <div
        id="coord-display"
        style={{
          position: 'absolute',
          bottom: '108px',
          left: '12px',
          zIndex: 10,
          pointerEvents: 'none',
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: '9px',
          color: 'rgba(240,246,252,0.7)',
          letterSpacing: '0.04em',
          background: 'rgba(13,17,23,0.75)',
          borderRadius: '3px',
          padding: '3px 8px',
        }}
      >
        34.0522° N, 118.2437° W
      </div>

      {/* Minimal compass rose */}
      <div
        style={{
          position: 'absolute',
          top: '52px',
          right: '52px',
          zIndex: 10,
          pointerEvents: 'none',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1px',
        }}
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <polygon points="7,0 5,7 7,5 9,7" fill="white" opacity="0.8" />
          <polygon points="7,14 5,7 7,9 9,7" fill="white" opacity="0.3" />
        </svg>
        <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.8)', fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, lineHeight: 1 }}>N</span>
      </div>
    </div>
  );
}

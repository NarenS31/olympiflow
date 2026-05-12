import { useEffect, useRef, useCallback } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useSimulationStore } from '../../stores/simulationStore';
import { LA28_VENUES } from '../../data/venues';
import { LA_ARTERIES, buildArteryGeoJSON } from '../../data/arteries';
import { generateHeatmapGeoJSON, generateZoneCongestionGeoJSON } from '../../utils/simulation';
import type { Venue } from '../../types';

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

function initCityLights(): CityLight[] {
  const lights: CityLight[] = [];
  // Grid over the LA basin with jitter so it looks organic, not perfectly aligned
  const lngMin = -118.555, lngMax = -118.095;
  const latMin = 33.745,  latMax = 34.220;
  const step = 0.0055; // ~600m grid spacing
  for (let lng = lngMin; lng <= lngMax; lng += step) {
    for (let lat = latMin; lat <= latMax; lat += step) {
      if (Math.random() > 0.38) continue; // sparse — not every block visible
      const p = LIGHT_PALETTE[Math.floor(Math.random() * LIGHT_PALETTE.length)];
      lights.push({
        lng: lng + (Math.random() - 0.5) * step * 0.9,
        lat: lat + (Math.random() - 0.5) * step * 0.9,
        size: 0.55 + Math.random() * 1.05,
        color: p.color,
        glowColor: p.glow,
        phase: Math.random() * Math.PI * 2,
        speed: 0.00022 + Math.random() * 0.00055, // very slow pulse
        baseOpacity: 0.10 + Math.random() * 0.26,
      });
    }
  }
  return lights;
}

// ── Venue icons ──────────────────────────────────────────────────────────────
const VENUE_ICONS: Record<string, string> = {
  stadium: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 11C3 7 7 4 12 4s9 3 9 7"/><path d="M3 11v6h18v-6"/>
    <line x1="7" y1="17" x2="7" y2="11"/><line x1="17" y1="17" x2="17" y2="11"/><line x1="12" y1="17" x2="12" y2="11"/>
  </svg>`,
  arena: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
    <path d="M2 13 Q12 3 22 13"/><rect x="2" y="13" width="20" height="5" rx="1"/>
    <line x1="12" y1="3" x2="12" y2="13"/><line x1="7" y1="18" x2="7" y2="13"/><line x1="17" y1="18" x2="17" y2="13"/>
  </svg>`,
  outdoor: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polygon points="12 2 3 16 21 16"/><line x1="12" y1="16" x2="12" y2="22"/><line x1="9" y1="22" x2="15" y2="22"/>
  </svg>`,
  aquatic: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
    <path d="M2 12 Q5.5 7 9 12 Q12.5 17 16 12 Q19.5 7 23 12"/>
    <path d="M2 18 Q5.5 13 9 18 Q12.5 23 16 18 Q19.5 13 23 18"/>
  </svg>`,
  velodrome: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
    <circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3"/>
    <line x1="12" y1="3" x2="12" y2="9"/><line x1="12" y1="15" x2="12" y2="21"/>
  </svg>`,
  flame: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C9.5 5.5 8 8.5 9 11.5c.5 1.5 1.5 2.5 2.5 3C11 13 11.5 11.5 13 11c-.5 3 1 5 2 6.5C16 15.5 16 13 14.5 11c1.5 1 2.5 3 2.5 5 1-1.5 1.5-3.5 1-5.5C17 7 14.5 4 12 2z"/>
    <path d="M12 20c-1.5 0-2.5-1-2.5-2.5S10.5 15 12 14c1.5 1 2.5 2 2.5 3.5S13.5 20 12 20z"/>
  </svg>`,
};

function getVenueIcon(venue: Venue) {
  if (venue.isOpeningClosing) return VENUE_ICONS.flame;
  return VENUE_ICONS[venue.type] ?? VENUE_ICONS.stadium;
}

function getVenueColors(venue: Venue) {
  if (venue.isOpeningClosing) return { border: '#b45309', glow: 'rgba(180,83,9,0.4)',   bg: 'rgba(180,83,9,0.12)'  };
  switch (venue.type) {
    case 'arena':   return { border: '#7c3aed', glow: 'rgba(124,58,237,0.35)', bg: 'rgba(124,58,237,0.10)' };
    case 'outdoor': return { border: '#16a34a', glow: 'rgba(22,163,74,0.35)',  bg: 'rgba(22,163,74,0.10)'  };
    case 'aquatic': return { border: '#1d4ed8', glow: 'rgba(29,78,216,0.35)',  bg: 'rgba(29,78,216,0.10)'  };
    default:        return { border: '#0891b2', glow: 'rgba(8,145,178,0.35)',  bg: 'rgba(8,145,178,0.10)'  };
  }
}

function createVenueMarkerElement(venue: Venue): HTMLElement {
  const { border, glow, bg } = getVenueColors(venue);
  const wrapper = document.createElement('div');
  wrapper.style.cssText = `display:flex;flex-direction:column;align-items:center;cursor:pointer;filter:drop-shadow(0 4px 14px ${glow});transition:filter 0.2s ease;`;

  const body = document.createElement('div');
  body.style.cssText = `width:40px;height:40px;border-radius:50% 50% 50% 4px;border:2px solid ${border};background:rgba(8,12,24,0.94);display:flex;align-items:center;justify-content:center;color:${border};backdrop-filter:blur(8px);box-shadow:inset 0 1px 0 rgba(255,255,255,0.08),0 2px 10px rgba(0,0,0,0.6);position:relative;overflow:hidden;transition:transform 0.18s ease,box-shadow 0.18s ease;transform-origin:bottom center;`;

  const glowLayer = document.createElement('div');
  glowLayer.style.cssText = `position:absolute;inset:0;border-radius:inherit;background:${bg};`;
  body.appendChild(glowLayer);
  body.insertAdjacentHTML('beforeend', getVenueIcon(venue));

  const tip = document.createElement('div');
  tip.style.cssText = `width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-top:8px solid ${border};margin-top:-1px;`;

  wrapper.appendChild(body);
  wrapper.appendChild(tip);

  wrapper.addEventListener('mouseenter', () => {
    wrapper.style.filter = `drop-shadow(0 8px 28px ${glow})`;
    body.style.transform = 'scale(1.22)';
    body.style.boxShadow = `inset 0 1px 0 rgba(255,255,255,0.12),0 4px 20px rgba(0,0,0,0.7),0 0 0 1px ${border}`;
  });
  wrapper.addEventListener('mouseleave', () => {
    wrapper.style.filter = `drop-shadow(0 4px 14px ${glow})`;
    body.style.transform = 'scale(1)';
    body.style.boxShadow = `inset 0 1px 0 rgba(255,255,255,0.08),0 2px 10px rgba(0,0,0,0.6)`;
  });
  return wrapper;
}

function createVenuePopupHTML(venue: Venue): string {
  const { border } = getVenueColors(venue);
  const sports = venue.sports
    .map((s) => `<span style="display:inline-block;margin:2px 4px 2px 0;padding:2px 8px;border-radius:20px;border:1px solid #1e293b;font-size:10px;color:#64748b;">${s}</span>`)
    .join('');
  return `<div style="font-family:'Inter',sans-serif;color:#e2e8f0;padding:14px;min-width:210px;">
    <div style="font-size:13px;font-weight:700;color:${border};margin-bottom:4px;">${venue.name}</div>
    <div style="font-size:10px;color:#475569;margin-bottom:10px;">${venue.neighborhood}&nbsp;·&nbsp;${venue.capacity.toLocaleString()} seats</div>
    <div>${sports}</div>
  </div>`;
}

// ── Component ────────────────────────────────────────────────────────────────
export function OlympiMap() {
  const mapContainer    = useRef<HTMLDivElement>(null);
  const canvasRef       = useRef<HTMLCanvasElement>(null);
  const map             = useRef<maplibregl.Map | null>(null);
  const markersRef      = useRef<maplibregl.Marker[]>([]);
  const isLoaded        = useRef(false);
  const animRafRef      = useRef<number>(0);
  const particleRafRef  = useRef<number>(0);
  const particlesRef    = useRef<Particle[]>([]);
  const cityLightsRef   = useRef<CityLight[]>([]);

  const venueSurges     = useSimulationStore((s) => s.venueSurges);
  const globalIntensity = useSimulationStore((s) => s.globalIntensity);
  const timeOfDay       = useSimulationStore((s) => s.timeOfDay);
  const layers          = useSimulationStore((s) => s.layers);
  const transitData     = useSimulationStore((s) => s.transitData);
  const heatmapBaseData = useSimulationStore((s) => s.heatmapBaseData);
  const selectVenue     = useSimulationStore((s) => s.selectVenue);

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
      pitch: 20,
    });

    map.current.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

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
          // Softer intensity — less aggressive punch at all zooms
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 8, 0.6, 14, 1.8],
          // Single warm ramp — no blue/teal that clashes with zone greens
          'heatmap-color': ['interpolate', ['linear'], ['heatmap-density'],
            0,    'rgba(0,0,0,0)',
            0.20, 'rgba(120,40,8,0.28)',
            0.45, 'rgba(160,55,10,0.50)',
            0.70, 'rgba(185,40,12,0.68)',
            1.0,  'rgba(180,24,18,0.82)',
          ],
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 8, 16, 14, 36],
          'heatmap-opacity': 0.55,
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
            0, 0.0, 0.12, 0.08, 0.40, 0.22, 1.0, 0.40],
        },
      });
      m.addLayer({
        id: 'zone-fill', type: 'fill', source: 'zones-source',
        paint: {
          'fill-color': ['interpolate', ['linear'], ['get', 'congestion'],
            0, '#166534', 0.28, '#854d0e', 0.50, '#9a3412', 0.72, '#991b1b', 1.0, '#7f1d1d'],
          'fill-opacity': ['interpolate', ['linear'], ['get', 'congestion'],
            0, 0.0, 0.08, 0.03, 0.25, 0.10, 0.50, 0.18, 0.75, 0.27, 1.0, 0.36],
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

      // ── Venue markers ─────────────────────────────────────────────────────
      for (const venue of LA28_VENUES) {
        const el = createVenueMarkerElement(venue);
        const popup = new maplibregl.Popup({ offset: [0, -48], closeButton: false, className: 'olympi-popup', maxWidth: '260px' })
          .setHTML(createVenuePopupHTML(venue));
        const marker = new maplibregl.Marker({ element: el, anchor: 'bottom' })
          .setLngLat([venue.lng, venue.lat]).setPopup(popup).addTo(m);
        el.addEventListener('click', () => selectVenue(venue.id));
        markersRef.current.push(marker);
      }

      // ── Global styles ─────────────────────────────────────────────────────
      const style = document.createElement('style');
      style.textContent = `
        .olympi-popup .maplibregl-popup-content {
          background:rgba(8,12,24,0.97);border:1px solid #182438;border-radius:12px;
          padding:0;box-shadow:0 8px 40px rgba(0,0,0,0.8),0 0 0 1px rgba(255,255,255,0.04);backdrop-filter:blur(16px);
        }
        .olympi-popup .maplibregl-popup-tip { display:none; }
        .maplibregl-ctrl-attrib { display:none !important; }
        .maplibregl-ctrl-group {
          background:rgba(8,12,24,0.94)!important;border:1px solid #182438!important;
          border-radius:10px!important;overflow:hidden;backdrop-filter:blur(12px);box-shadow:0 4px 20px rgba(0,0,0,0.5);
        }
        .maplibregl-ctrl-group button {
          background:transparent!important;color:#3d5270!important;border:none!important;width:34px!important;height:34px!important;
        }
        .maplibregl-ctrl-group button:hover{background:#182438!important;color:#e2e8f0!important;}
        .maplibregl-ctrl-group button+button{border-top:1px solid #182438!important;}
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
      ?.setData(generateZoneCongestionGeoJSON(venueSurges, globalIntensity, timeOfDay));
  }, [venueSurges, globalIntensity, timeOfDay]);

  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    const pts = getBasePoints();
    const geojson = generateHeatmapGeoJSON(pts, venueSurges, globalIntensity, timeOfDay);
    (map.current.getSource('heatmap-source') as maplibregl.GeoJSONSource)?.setData(geojson);
  }, [venueSurges, globalIntensity, timeOfDay, heatmapBaseData, getBasePoints]);

  useEffect(() => {
    if (!isLoaded.current || !map.current || !transitData) return;
    (map.current.getSource('transit-source') as maplibregl.GeoJSONSource)?.setData(transitData);
  }, [transitData]);

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
    markersRef.current.forEach((mk) => { mk.getElement().style.display = layers.venues ? 'flex' : 'none'; });
  }, [layers]);

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
    </div>
  );
}

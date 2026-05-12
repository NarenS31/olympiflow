import { useEffect, useRef, useCallback } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useSimulationStore } from '../../stores/simulationStore';
import { LA28_VENUES } from '../../data/venues';
import { buildArteryGeoJSON } from '../../data/arteries';
import { generateHeatmapGeoJSON, generateZoneCongestionGeoJSON } from '../../utils/simulation';
import type { Venue } from '../../types';

const LA_CENTER: [number, number] = [-118.2437, 34.0522];
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

// Animated dasharray sequence — creates "ant march" flowing transit lines
const DASH_SEQUENCE = [
  [0, 4, 3],
  [0.5, 4, 2.5],
  [1, 4, 2],
  [1.5, 4, 1.5],
  [2, 4, 1],
  [2.5, 4, 0.5],
  [3, 4, 0],
  [3.5, 3.5, 0],
];

const VENUE_ICONS: Record<string, string> = {
  stadium: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 11C3 7 7 4 12 4s9 3 9 7"/>
    <path d="M3 11v6h18v-6"/>
    <line x1="7" y1="17" x2="7" y2="11"/>
    <line x1="17" y1="17" x2="17" y2="11"/>
    <line x1="12" y1="17" x2="12" y2="11"/>
  </svg>`,
  arena: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
    <path d="M2 13 Q12 3 22 13"/>
    <rect x="2" y="13" width="20" height="5" rx="1"/>
    <line x1="12" y1="3" x2="12" y2="13"/>
    <line x1="7" y1="18" x2="7" y2="13"/>
    <line x1="17" y1="18" x2="17" y2="13"/>
  </svg>`,
  outdoor: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polygon points="12 2 3 16 21 16"/>
    <line x1="12" y1="16" x2="12" y2="22"/>
    <line x1="9" y1="22" x2="15" y2="22"/>
  </svg>`,
  aquatic: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
    <path d="M2 12 Q5.5 7 9 12 Q12.5 17 16 12 Q19.5 7 23 12"/>
    <path d="M2 18 Q5.5 13 9 18 Q12.5 23 16 18 Q19.5 13 23 18"/>
  </svg>`,
  velodrome: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
    <circle cx="12" cy="12" r="9"/>
    <circle cx="12" cy="12" r="3"/>
    <line x1="12" y1="3" x2="12" y2="9"/>
    <line x1="12" y1="15" x2="12" y2="21"/>
  </svg>`,
  flame: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C9.5 5.5 8 8.5 9 11.5c.5 1.5 1.5 2.5 2.5 3C11 13 11.5 11.5 13 11c-.5 3 1 5 2 6.5C16 15.5 16 13 14.5 11c1.5 1 2.5 3 2.5 5 1-1.5 1.5-3.5 1-5.5C17 7 14.5 4 12 2z"/>
    <path d="M12 20c-1.5 0-2.5-1-2.5-2.5S10.5 15 12 14c1.5 1 2.5 2 2.5 3.5S13.5 20 12 20z"/>
  </svg>`,
};

function getVenueIcon(venue: Venue): string {
  if (venue.isOpeningClosing) return VENUE_ICONS.flame;
  return VENUE_ICONS[venue.type] ?? VENUE_ICONS.stadium;
}

function getVenueColors(venue: Venue) {
  if (venue.isOpeningClosing) return { border: '#b45309', glow: 'rgba(180,83,9,0.4)',   bg: 'rgba(180,83,9,0.12)'  };
  switch (venue.type) {
    case 'arena':    return { border: '#7c3aed', glow: 'rgba(124,58,237,0.35)', bg: 'rgba(124,58,237,0.10)' };
    case 'outdoor':  return { border: '#16a34a', glow: 'rgba(22,163,74,0.35)',  bg: 'rgba(22,163,74,0.10)'  };
    case 'aquatic':  return { border: '#1d4ed8', glow: 'rgba(29,78,216,0.35)',  bg: 'rgba(29,78,216,0.10)'  };
    default:         return { border: '#0891b2', glow: 'rgba(8,145,178,0.35)',  bg: 'rgba(8,145,178,0.10)'  };
  }
}

function createVenueMarkerElement(venue: Venue): HTMLElement {
  const { border, glow, bg } = getVenueColors(venue);
  const icon = getVenueIcon(venue);

  const wrapper = document.createElement('div');
  wrapper.style.cssText = `
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: pointer;
    filter: drop-shadow(0 4px 14px ${glow});
    transition: filter 0.2s ease;
  `;

  const body = document.createElement('div');
  body.style.cssText = `
    width: 40px;
    height: 40px;
    border-radius: 50% 50% 50% 4px;
    border: 2px solid ${border};
    background: rgba(8,12,24,0.94);
    display: flex;
    align-items: center;
    justify-content: center;
    color: ${border};
    backdrop-filter: blur(8px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 2px 10px rgba(0,0,0,0.6);
    position: relative;
    overflow: hidden;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    transform-origin: bottom center;
  `;

  const glowLayer = document.createElement('div');
  glowLayer.style.cssText = `
    position: absolute; inset: 0; border-radius: inherit;
    background: ${bg};
  `;
  body.appendChild(glowLayer);
  body.insertAdjacentHTML('beforeend', icon);

  const tip = document.createElement('div');
  tip.style.cssText = `
    width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 8px solid ${border};
    margin-top: -1px;
  `;

  wrapper.appendChild(body);
  wrapper.appendChild(tip);

  wrapper.addEventListener('mouseenter', () => {
    wrapper.style.filter = `drop-shadow(0 8px 28px ${glow})`;
    body.style.transform = 'scale(1.22)';
    body.style.boxShadow = `inset 0 1px 0 rgba(255,255,255,0.12), 0 4px 20px rgba(0,0,0,0.7), 0 0 0 1px ${border}`;
  });
  wrapper.addEventListener('mouseleave', () => {
    wrapper.style.filter = `drop-shadow(0 4px 14px ${glow})`;
    body.style.transform = 'scale(1)';
    body.style.boxShadow = `inset 0 1px 0 rgba(255,255,255,0.08), 0 2px 10px rgba(0,0,0,0.6)`;
  });

  return wrapper;
}

function createVenuePopupHTML(venue: Venue): string {
  const { border } = getVenueColors(venue);
  const sportsHTML = venue.sports
    .map((s) => `<span style="display:inline-block;margin:2px 4px 2px 0;padding:2px 8px;border-radius:20px;border:1px solid #1e293b;font-size:10px;color:#64748b;">${s}</span>`)
    .join('');
  return `
    <div style="font-family:'Inter',sans-serif;color:#e2e8f0;padding:14px;min-width:210px;">
      <div style="font-size:13px;font-weight:700;color:${border};margin-bottom:4px;">${venue.name}</div>
      <div style="font-size:10px;color:#475569;margin-bottom:10px;">
        ${venue.neighborhood}&nbsp;&nbsp;·&nbsp;&nbsp;${venue.capacity.toLocaleString()} seats
      </div>
      <div>${sportsHTML}</div>
    </div>
  `;
}

// Congestion color expression shared between artery layers
const CONGESTION_COLOR_EXPR = [
  'interpolate', ['linear'], ['get', 'congestion'],
  0,    '#166534',
  0.25, '#854d0e',
  0.50, '#9a3412',
  0.75, '#991b1b',
  1.0,  '#7f1d1d',
];

export function OlympiMap() {
  const mapContainer   = useRef<HTMLDivElement>(null);
  const map            = useRef<maplibregl.Map | null>(null);
  const markersRef     = useRef<maplibregl.Marker[]>([]);
  const isLoaded       = useRef(false);
  const dashStepRef    = useRef(0);
  const animRafRef     = useRef<number>(0);

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

  // ── Map init ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (map.current || !mapContainer.current) return;

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

      // ── Artery road glow (major LA freeways) ────────────────────────────
      m.addSource('arteries-source', {
        type: 'geojson',
        data: buildArteryGeoJSON() as GeoJSON.FeatureCollection,
      });

      // Wide outer bloom
      m.addLayer({
        id: 'artery-bloom',
        type: 'line',
        source: 'arteries-source',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': '#9a3412',
          'line-width': 18,
          'line-opacity': 0,
          'line-blur': 6,
        },
      });

      // Mid glow
      m.addLayer({
        id: 'artery-glow',
        type: 'line',
        source: 'arteries-source',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': '#7f1d1d',
          'line-width': 8,
          'line-opacity': 0,
          'line-blur': 3,
        },
      });

      // Core artery line
      m.addLayer({
        id: 'artery-core',
        type: 'line',
        source: 'arteries-source',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': '#9a3412',
          'line-width': 2,
          'line-opacity': 0,
        },
      });

      // ── Zone congestion overlay ─────────────────────────────────────────
      m.addSource('zones-source', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });

      m.addLayer({
        id: 'zone-fill',
        type: 'fill',
        source: 'zones-source',
        paint: {
          'fill-color': [
            'interpolate', ['linear'], ['get', 'congestion'],
            0,    '#166534',
            0.28, '#854d0e',
            0.50, '#9a3412',
            0.72, '#991b1b',
            1.0,  '#7f1d1d',
          ],
          'fill-opacity': [
            'interpolate', ['linear'], ['get', 'congestion'],
            0, 0.0, 0.08, 0.04, 0.25, 0.12, 0.50, 0.22, 0.75, 0.32, 1.0, 0.42,
          ],
        },
      });

      m.addLayer({
        id: 'zone-border',
        type: 'line',
        source: 'zones-source',
        paint: {
          'line-color': [
            'interpolate', ['linear'], ['get', 'congestion'],
            0,   '#166534',
            0.5, '#9a3412',
            1.0, '#991b1b',
          ],
          'line-width': 0.8,
          'line-opacity': [
            'interpolate', ['linear'], ['get', 'congestion'],
            0, 0.0, 0.15, 0.20, 1.0, 0.55,
          ],
        },
      });

      // ── Heatmap — more aggressive ramp ─────────────────────────────────
      m.addSource('heatmap-source', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });

      m.addLayer({
        id: 'traffic-heatmap',
        type: 'heatmap',
        source: 'heatmap-source',
        paint: {
          'heatmap-weight': ['interpolate', ['linear'], ['get', 'weight'], 0, 0, 1, 1],
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 8, 1.2, 14, 3.5],
          'heatmap-color': [
            'interpolate', ['linear'], ['heatmap-density'],
            0,    'rgba(0,0,0,0)',
            0.10, 'rgba(29,78,216,0.4)',
            0.30, 'rgba(8,145,178,0.65)',
            0.50, 'rgba(180,83,9,0.82)',
            0.70, 'rgba(185,28,28,0.92)',
            1.0,  'rgba(127,29,29,1)',
          ],
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 8, 18, 14, 42],
          'heatmap-opacity': 0.88,
        },
      });

      // ── Transit routes ──────────────────────────────────────────────────
      m.addSource('transit-source', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });

      // Glow casing
      m.addLayer({
        id: 'transit-casing',
        type: 'line',
        source: 'transit-source',
        layout: { 'line-cap': 'round' },
        paint: { 'line-color': '#000', 'line-width': 6, 'line-opacity': 0.4 },
      });

      // Base route line
      m.addLayer({
        id: 'transit-routes',
        type: 'line',
        source: 'transit-source',
        layout: { 'line-cap': 'round' },
        paint: {
          'line-color': ['coalesce', ['get', 'color'], '#6d28d9'],
          'line-width': 2.5,
          'line-opacity': 0.55,
        },
      });

      // Animated flow layer on top of routes
      m.addLayer({
        id: 'transit-flow',
        type: 'line',
        source: 'transit-source',
        layout: { 'line-cap': 'round' },
        paint: {
          'line-color': ['coalesce', ['get', 'color'], '#7c3aed'],
          'line-width': 2,
          'line-dasharray': [0, 4, 3],
          'line-opacity': 0.85,
        },
      });

      // ── Start animated dash loop ────────────────────────────────────────
      let lastStep = -1;
      let lastTs = 0;
      const FRAME_MS = 80; // ~12 fps feels smooth enough for dashes

      function animateDash(ts: number) {
        if (ts - lastTs >= FRAME_MS) {
          const step = Math.floor(ts / FRAME_MS) % DASH_SEQUENCE.length;
          if (step !== lastStep && m.getLayer('transit-flow')) {
            m.setPaintProperty('transit-flow', 'line-dasharray', DASH_SEQUENCE[step]);
            dashStepRef.current = step;
            lastStep = step;
          }
          lastTs = ts;
        }
        animRafRef.current = requestAnimationFrame(animateDash);
      }
      animRafRef.current = requestAnimationFrame(animateDash);

      // ── Venue markers ───────────────────────────────────────────────────
      for (const venue of LA28_VENUES) {
        const el = createVenueMarkerElement(venue);
        const popup = new maplibregl.Popup({
          offset: [0, -48],
          closeButton: false,
          className: 'olympi-popup',
          maxWidth: '260px',
        }).setHTML(createVenuePopupHTML(venue));

        const marker = new maplibregl.Marker({ element: el, anchor: 'bottom' })
          .setLngLat([venue.lng, venue.lat])
          .setPopup(popup)
          .addTo(m);

        el.addEventListener('click', () => selectVenue(venue.id));
        markersRef.current.push(marker);
      }

      // ── Global popup / control styles ───────────────────────────────────
      const style = document.createElement('style');
      style.textContent = `
        .olympi-popup .maplibregl-popup-content {
          background: rgba(8,12,24,0.97);
          border: 1px solid #182438;
          border-radius: 12px;
          padding: 0;
          box-shadow: 0 8px 40px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.04);
          backdrop-filter: blur(16px);
        }
        .olympi-popup .maplibregl-popup-tip { display: none; }
        .maplibregl-ctrl-attrib { display: none !important; }
        .maplibregl-ctrl-group {
          background: rgba(8,12,24,0.94) !important;
          border: 1px solid #182438 !important;
          border-radius: 10px !important;
          overflow: hidden;
          backdrop-filter: blur(12px);
          box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .maplibregl-ctrl-group button {
          background: transparent !important;
          color: #3d5270 !important;
          border: none !important;
          width: 34px !important;
          height: 34px !important;
        }
        .maplibregl-ctrl-group button:hover { background: #182438 !important; color: #e2e8f0 !important; }
        .maplibregl-ctrl-group button + button { border-top: 1px solid #182438 !important; }
      `;
      document.head.appendChild(style);
    });

    return () => {
      cancelAnimationFrame(animRafRef.current);
      map.current?.remove();
      map.current = null;
      isLoaded.current = false;
      markersRef.current = [];
    };
  }, [selectVenue]);

  // ── Update zone congestion overlay ─────────────────────────────────────
  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    const src = map.current.getSource('zones-source') as maplibregl.GeoJSONSource;
    src?.setData(generateZoneCongestionGeoJSON(venueSurges, globalIntensity, timeOfDay));
  }, [venueSurges, globalIntensity, timeOfDay]);

  // ── Update heatmap ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    const pts = getBasePoints();
    const geojson = generateHeatmapGeoJSON(pts, venueSurges, globalIntensity, timeOfDay);
    (map.current.getSource('heatmap-source') as maplibregl.GeoJSONSource)?.setData(geojson);
  }, [venueSurges, globalIntensity, timeOfDay, heatmapBaseData, getBasePoints]);

  // ── Update transit ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!isLoaded.current || !map.current || !transitData) return;
    (map.current.getSource('transit-source') as maplibregl.GeoJSONSource)?.setData(transitData);
  }, [transitData]);

  // ── Update artery glow based on globalIntensity ─────────────────────────
  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    const m = map.current;
    if (!m.getLayer('artery-bloom')) return;

    const bloomOpacity = Math.max(0, (globalIntensity - 0.2) * 0.22);
    const glowOpacity  = Math.max(0, (globalIntensity - 0.2) * 0.38);
    const coreOpacity  = Math.max(0, (globalIntensity - 0.15) * 0.65);

    m.setPaintProperty('artery-bloom', 'line-opacity', bloomOpacity);
    m.setPaintProperty('artery-glow',  'line-opacity', glowOpacity);
    m.setPaintProperty('artery-core',  'line-opacity', coreOpacity);

    // Shift color toward red as intensity rises
    const color = globalIntensity > 0.7 ? '#991b1b' : globalIntensity > 0.45 ? '#9a3412' : '#854d0e';
    m.setPaintProperty('artery-bloom', 'line-color', color);
    m.setPaintProperty('artery-glow',  'line-color', color);
    m.setPaintProperty('artery-core',  'line-color', color);
  }, [globalIntensity]);

  // ── Layer visibility ────────────────────────────────────────────────────
  useEffect(() => {
    if (!isLoaded.current || !map.current) return;
    const m = map.current;
    const vis = (id: string, on: boolean) => {
      if (m.getLayer(id)) m.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none');
    };
    vis('zone-fill',    layers.heatmap);
    vis('zone-border',  layers.heatmap);
    vis('traffic-heatmap', layers.heatmap);
    vis('artery-bloom', layers.heatmap);
    vis('artery-glow',  layers.heatmap);
    vis('artery-core',  layers.heatmap);
    vis('transit-routes', layers.transit);
    vis('transit-casing', layers.transit);
    vis('transit-flow',   layers.transit);
    markersRef.current.forEach((mk) => {
      mk.getElement().style.display = layers.venues ? 'flex' : 'none';
    });
  }, [layers]);

  return (
    <div ref={mapContainer} className="w-full h-full" style={{ background: '#04080f' }} />
  );
}

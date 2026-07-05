import { useEffect } from 'react';
import { OlympiMap } from './components/Map/OlympiMap';
import { SimulationSidebar } from './components/Sidebar/SimulationSidebar';
import { Header } from './components/UI/Header';
import { MetricsPanel } from './components/Dashboard/MetricsPanel';
import { TimelineSlider } from './components/Dashboard/TimelineSlider';
import { MapLegend } from './components/UI/MapLegend';
import { useSimulationData, useSimulationTick } from './hooks/useSimulation';
import { useMLPredictions } from './hooks/useMLPredictions';

function App() {
  useSimulationData();
  useSimulationTick();
  useMLPredictions();

  useEffect(() => {
    const handler = (e: WheelEvent) => {
      if (e.ctrlKey) e.preventDefault();
    };
    document.addEventListener('wheel', handler, { passive: false });
    return () => document.removeEventListener('wheel', handler);
  }, []);

  return (
    <div className="fixed inset-0 overflow-hidden" style={{ background: '#0D1117' }}>
      {/* Full-screen map */}
      <div className="absolute inset-0" style={{ bottom: '24px' }}>
        <OlympiMap />
      </div>

      {/* Header overlay */}
      <Header />

      {/* Left simulation controls */}
      <SimulationSidebar />

      {/* Right metrics panel */}
      <MetricsPanel />

      {/* Bottom timeline */}
      <TimelineSlider />

      {/* Map legend */}
      <MapLegend />

      {/* Bottom status bar */}
      <div
        style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          height: '24px',
          background: 'rgba(18,19,22,0.70)',
          borderTop: '0.5px solid rgba(255,255,255,0.10)',
          backdropFilter: 'blur(40px) saturate(160%)',
          WebkitBackdropFilter: 'blur(40px) saturate(160%)',
          boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 14px',
          zIndex: 50,
          pointerEvents: 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#30D158', display: 'inline-block', flexShrink: 0 }} />
          <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.25)', letterSpacing: '0.04em' }}>
            System Active
          </span>
        </div>
        <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.2)', letterSpacing: '0.02em' }}>
          LADOT · LA Open Data Portal · LA28 Dataset
        </div>
        <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.2)', fontFamily: "'SF Mono', ui-monospace, monospace", letterSpacing: '0.02em' }}>
          24 Jun 2026
        </div>
      </div>
    </div>
  );
}

export default App;

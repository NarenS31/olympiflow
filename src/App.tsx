import { useEffect } from 'react';
import { OlympiMap } from './components/Map/OlympiMap';
import { SimulationSidebar } from './components/Sidebar/SimulationSidebar';
import { Header } from './components/UI/Header';
import { MetricsPanel } from './components/Dashboard/MetricsPanel';
import { TimelineSlider } from './components/Dashboard/TimelineSlider';
import { MapLegend } from './components/UI/MapLegend';
import { useSimulationData, useSimulationTick } from './hooks/useSimulation';

function App() {
  useSimulationData();
  useSimulationTick();

  // Prevent browser zoom from interfering with sliders
  useEffect(() => {
    const handler = (e: WheelEvent) => {
      if (e.ctrlKey) e.preventDefault();
    };
    document.addEventListener('wheel', handler, { passive: false });
    return () => document.removeEventListener('wheel', handler);
  }, []);

  return (
    <div className="fixed inset-0 overflow-hidden bg-surface-900">
      {/* Full-screen map */}
      <div className="absolute inset-0">
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
    </div>
  );
}

export default App;

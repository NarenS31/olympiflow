# OlympiFlow — Setup Guide

## Prerequisites

- **Node.js** 20+ (install via https://nodejs.org or `brew install node`)
- **Python** 3.9+ (already available)
- **pip** (already available)

---

## 1. Install Node.js (if not already installed)

```bash
# Option A — Official installer (recommended)
# Download from https://nodejs.org/en/download

# Option B — Homebrew
brew install node

# Option C — nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
nvm install 20
nvm use 20
```

---

## 2. Frontend Setup

```bash
cd /Users/narensara/Desktop/OlympiFlow
npm install
npm run dev          # starts at http://localhost:3000
```

---

## 3. Backend Setup (in a separate terminal)

```bash
cd /Users/narensara/Desktop/OlympiFlow/backend

# Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --reload --port 8000
```

The frontend Vite dev server proxies `/api/*` → `http://localhost:8000`, so both
must run simultaneously for full functionality.  
If the backend is not running, the map still loads using empty fallback data.

---

## 4. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/traffic/heatmap` | GeoJSON heatmap from collision data |
| GET | `/api/traffic/stats` | LADOT traffic count aggregates |
| GET | `/api/transit/routes` | Downtown DASH routes as GeoJSON |
| GET | `/api/transit/parking` | Parking occupancy summary |
| GET | `/api/venues/` | List of all LA28 venues |
| GET | `/api/venues/geojson` | Venues as GeoJSON |
| POST | `/api/simulation/step` | Run simulation step, get metrics |

---

## 5. Application Features

### Map Layers (toggle in sidebar)
- **Traffic Heatmap** — Real collision hotspots from LAPD data (4000 points)
- **Olympic Venues** — 12 confirmed LA28 venues with sport details
- **Transit Routes** — Downtown DASH A–F routes
- **Parking Density** — Estimated occupancy by zone

### Simulation Controls
- **Mode**: Baseline / Event Day / Crisis — presets global intensity
- **Global Intensity** — Master traffic pressure slider
- **Venue Crowd Surge** — Per-venue crowd injection (adds pressure rings on heatmap)
- **Timeline Slider** — Playable 24-hour time cycle
- **Quick Actions** — One-click traffic surge, mass venue event, reset

### Metrics Panel
- Average delay increase (BPR model)
- Congestion score (0–100%)
- Peak congestion zones count
- Affected transit routes
- Estimated persons affected
- Congestion-over-time chart (Recharts)

---

## 6. Architecture

```
Frontend (React 18 + TS + Vite + MapLibre)
  ├── OlympiMap.tsx      — MapLibre with heatmap, venue markers, transit overlay
  ├── SimulationStore    — Zustand state for all simulation parameters
  ├── useSimulation      — Tick loop + backend data loading
  └── simulation.ts      — BPR model, surge diffusion, metrics math

Backend (FastAPI + Python)
  ├── /traffic           — Collision heatmap, LADOT counts
  ├── /transit           — DASH routes, parking summary
  ├── /venues            — LA28 venue data
  └── /simulation        — Server-side metrics calculation
```

---

## 7. Data Files Used

| File | Usage |
|------|-------|
| `Traffic_Collision_Data_from_2010_to_Present_20260512.csv` | Base heatmap (lat/lng extracted) |
| `Downtown_DASH_Routes_20260512.csv` | Transit layer (MULTILINESTRING → GeoJSON) |
| `LADOT_Traffic_Counts_Summary_20260512.csv` | Stats panel (top intersections) |
| `LADOT_Parking_Meter_Occupancy_20260512.csv` | Parking occupancy aggregate |

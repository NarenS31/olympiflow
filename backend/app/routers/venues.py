from fastapi import APIRouter
from typing import Any

router = APIRouter(prefix="/venues", tags=["venues"])

LA28_VENUES = [
    {"id": "sofi", "name": "SoFi Stadium", "sports": ["Opening Ceremony", "Athletics"], "lng": -118.3378, "lat": 33.9533, "capacity": 70240, "neighborhood": "Inglewood"},
    {"id": "intuit-dome", "name": "Intuit Dome", "sports": ["Basketball", "Boxing"], "lng": -118.3416, "lat": 33.9583, "capacity": 18000, "neighborhood": "Inglewood"},
    {"id": "crypto-arena", "name": "Crypto.com Arena", "sports": ["Basketball 3x3", "Weightlifting"], "lng": -118.2674, "lat": 34.0430, "capacity": 20000, "neighborhood": "Downtown LA"},
    {"id": "la-coliseum", "name": "LA Memorial Coliseum", "sports": ["Athletics"], "lng": -118.2879, "lat": 34.0141, "capacity": 77500, "neighborhood": "Exposition Park"},
    {"id": "rose-bowl", "name": "Rose Bowl Stadium", "sports": ["Football (Soccer)"], "lng": -118.1677, "lat": 34.1614, "capacity": 92542, "neighborhood": "Pasadena"},
    {"id": "pauley", "name": "Pauley Pavilion", "sports": ["Gymnastics", "Trampoline"], "lng": -118.4436, "lat": 34.0702, "capacity": 13800, "neighborhood": "Westwood"},
    {"id": "bmo-stadium", "name": "BMO Stadium", "sports": ["Football (Soccer)"], "lng": -118.2840, "lat": 34.0131, "capacity": 22000, "neighborhood": "Exposition Park"},
    {"id": "long-beach-arena", "name": "Long Beach Arena", "sports": ["Volleyball"], "lng": -118.1887, "lat": 33.7697, "capacity": 13500, "neighborhood": "Long Beach"},
    {"id": "sepulveda-basin", "name": "Sepulveda Basin", "sports": ["Canoe Sprint", "Rowing"], "lng": -118.4760, "lat": 34.1820, "capacity": 25000, "neighborhood": "Encino"},
    {"id": "el-dorado", "name": "El Dorado Regional Park", "sports": ["Archery"], "lng": -118.0614, "lat": 33.8194, "capacity": 8000, "neighborhood": "Long Beach"},
    {"id": "dignity-health", "name": "Dignity Health Sports Park", "sports": ["Tennis", "Rugby Sevens"], "lng": -118.2614, "lat": 33.8636, "capacity": 30000, "neighborhood": "Carson"},
    {"id": "ucla-olympic", "name": "UCLA Olympic Village", "sports": ["Swimming", "Water Polo"], "lng": -118.4452, "lat": 34.0689, "capacity": 15000, "neighborhood": "Westwood"},
]


@router.get("/")
def list_venues() -> list[dict[str, Any]]:
    return LA28_VENUES


@router.get("/geojson")
def venues_geojson() -> dict[str, Any]:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [v["lng"], v["lat"]]},
            "properties": {k: val for k, val in v.items() if k not in ("lng", "lat")},
        }
        for v in LA28_VENUES
    ]
    return {"type": "FeatureCollection", "features": features}

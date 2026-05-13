from __future__ import annotations
import httpx
from typing import Any, Optional
from .crime_loader import load_crime_area_summary
from .data_loader import load_traffic_stats

OLLAMA_BASE = "http://localhost:11434"

_VENUE_CONTEXT = """LA 2028 Olympic Venues:
- SoFi Stadium (Inglewood, 70,240 seats) — opening/closing ceremony, soccer
- Crypto.com Arena (Downtown LA, 20,000 seats) — basketball, boxing
- Rose Bowl (Pasadena, 90,888 seats) — soccer
- Dignity Health Sports Park (Carson, 27,000 seats) — soccer
- Intuit Dome (Inglewood, 18,000 seats) — basketball, 3x3 basketball
- UCLA Pauley Pavilion (Westwood, 13,800 seats) — gymnastics, volleyball
- LA Memorial Coliseum (Exposition Park, 77,500 seats) — athletics/track
- Banc of California Stadium (Downtown LA, 22,000 seats) — rugby
- Long Beach Arena (Long Beach, 14,000 seats) — volleyball
- Sepulveda Basin (Van Nuys, outdoor) — canoe/kayak
- Balboa Sports Complex (Van Nuys, outdoor) — softball"""

_ARTERY_CONTEXT = """Key LA traffic corridors connecting Olympic venues:
- I-405 (San Diego Fwy): LAX ↔ Westwood/UCLA ↔ Inglewood (SoFi/Intuit) ↔ Long Beach
- I-110 (Harbor Fwy): Downtown LA ↔ Inglewood ↔ Carson
- I-10 (Santa Monica Fwy): Downtown ↔ West LA, east to I-605 Pasadena junction
- I-101 (Hollywood Fwy): Downtown ↔ Hollywood ↔ San Fernando Valley
- SR-134 (Ventura Fwy): Pasadena (Rose Bowl) ↔ Burbank
- SR-91 (Artesia Fwy): Long Beach ↔ eastern LA
- Wilshire Blvd: Westwood/UCLA east through Beverly Hills to Downtown
- Figueroa St / Vermont Ave: North-South corridors near Exposition Park & Coliseum
- Sepulveda Blvd: Van Nuys (Sepulveda Basin) south through Westwood to LAX
- Century Blvd: LAX to Inglewood (SoFi Stadium)"""

_ROUTING_STRATEGIES = """Olympic Traffic Management Strategies for LA 2028:
- Stagger event start times by 90+ minutes to prevent simultaneous stadium egress
- SoFi + Intuit Dome cluster (Inglewood): primary I-405/I-105/Century Blvd; satellite parking at Hollywood Park; overflow via Prairie Ave & Century Blvd
- Rose Bowl (Pasadena): Metro Gold/A Line park-and-ride is best; car traffic via SR-134 east or Arroyo Seco Pkwy; avoid I-210/SR-134 merge during peak
- Downtown cluster (Crypto.com, Banc of CA): Metro Red/Purple/Blue Lines primary; 7th St/Metro Center hub; vehicle traffic use I-110 ramps off 6th/9th St
- Exposition Park (Coliseum, Sports Arena): Metro Expo/E Line direct; vehicle routing via Vermont Ave south from I-10, Figueroa St north from I-110
- Westwood/UCLA (Pauley): Wilshire/Westwood shuttle corridor; avoid I-405 during evening; Metro Purple Line extension critical
- Sepulveda Basin (Van Nuys): US-101 to Balboa Ave exit; Burbank Blvd and Ventura Blvd as overflow cross-streets
- High-crime areas that intersect routes (77th St, Southeast, Central): daytime routing preferred; additional security/lighting corridor
- General: activate signal timing coordination on Olympic Blvd, Sepulveda Blvd, and Wilshire; deploy traffic control officers at 25 key intersections"""


def _build_docs() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []

    # Crime hotspots
    summary = load_crime_area_summary()
    if summary:
        lines = ["High-crime LA neighborhoods (2020–2024 LAPD data):"]
        for a in summary[:12]:
            tops = ", ".join(c["type"] for c in a["top_crimes"])
            lines.append(f"  - {a['area']}: {a['total_incidents']:,} incidents (top types: {tops})")
        docs.append({"id": "crime", "title": "Crime Hotspots by Neighborhood", "text": "\n".join(lines)})

    # Traffic counts
    try:
        stats = load_traffic_stats()
        if stats.get("topIntersections"):
            lines = [f"Highest daily traffic volume intersections (avg {stats['avgDailyVolume']:,.0f} vehicles/day):"]
            for item in stats["topIntersections"][:15]:
                lines.append(f"  - {item['street']}: {item['total']:,} vehicles")
            docs.append({"id": "traffic", "title": "Busiest Intersections by Daily Volume", "text": "\n".join(lines)})
    except Exception:
        pass

    docs.append({"id": "venues", "title": "LA 2028 Olympic Venue Locations & Capacity", "text": _VENUE_CONTEXT})
    docs.append({"id": "arteries", "title": "LA Major Traffic Corridors", "text": _ARTERY_CONTEXT})
    docs.append({"id": "strategies", "title": "Olympic Traffic Routing Strategies", "text": _ROUTING_STRATEGIES})

    return docs


def _score(query: str, text: str) -> float:
    """Simple keyword relevance — count term hits with mild TF saturation."""
    terms = [t for t in query.lower().split() if len(t) >= 3]
    lower = text.lower()
    score = 0.0
    for t in terms:
        hits = lower.count(t)
        if hits:
            score += 1.0 + 0.4 * min(hits - 1, 5)
    return score


def _retrieve(query: str, docs: list[dict[str, str]], top_k: int = 4) -> list[dict[str, str]]:
    ranked = sorted(docs, key=lambda d: _score(query, d["text"]), reverse=True)
    return ranked[:top_k]


async def get_traffic_advice(
    query: str,
    model: str = "llama3.2",
    simulation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """RAG pipeline: build context → retrieve top chunks → call Ollama → return answer."""
    docs = _build_docs()
    relevant = _retrieve(query, docs)

    context_block = "\n\n---\n\n".join(
        f"[{d['title']}]\n{d['text']}" for d in relevant
    )

    sim_line = ""
    if simulation_context:
        intensity = simulation_context.get("globalIntensity", 0)
        mode = simulation_context.get("mode", "baseline")
        tod = simulation_context.get("timeOfDay", 8)
        sim_line = (
            f"\nCurrent simulation: mode={mode}, "
            f"traffic intensity={intensity:.0%}, "
            f"time of day={tod:.1f}h\n"
        )

    prompt = (
        "You are an expert Olympic traffic engineer for the 2028 Los Angeles Games.\n"
        "Give specific, actionable advice using real street names, freeways, and venue names.\n"
        f"{sim_line}\n"
        "Context data (from LA city datasets and Olympic planning documents):\n\n"
        f"{context_block}\n\n"
        f"Question: {query}\n\n"
        "Provide a clear, concise answer in 3–6 bullet points with concrete routing recommendations:"
    )

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip()
    except httpx.ConnectError:
        answer = (
            "Cannot reach Ollama. Make sure it is running:\n"
            "  ollama serve\n"
            "Then pull a model:\n"
            "  ollama pull llama3.2"
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            answer = (
                f"Model '{model}' not found in Ollama. Pull it first:\n"
                f"  ollama pull {model}"
            )
        else:
            answer = f"Ollama returned HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        answer = f"Error: {e}"

    return {
        "answer": answer,
        "context_used": [d["title"] for d in relevant],
        "model": model,
    }

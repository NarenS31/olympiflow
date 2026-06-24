"""
Run this script while the OlympiFlow backend is running locally:

    cd backend && uvicorn app.main:app --reload

Then in a second terminal from the project root:

    python run_congestion_stats.py
"""
import json
import urllib.request
import urllib.error

URL = "http://127.0.0.1:8000/api/congestion-stats"


def fetch() -> dict:
    try:
        with urllib.request.urlopen(URL, timeout=5) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not reach {URL}\n"
            "Make sure the backend is running:  cd backend && uvicorn app.main:app --reload\n"
            f"Error: {e}"
        )


def print_table(data: dict) -> None:
    b = data["baseline"]
    i = data["intervention"]

    COL = 34
    W = 18

    def row(label, b_val, i_val, fmt=str):
        print(f"  {label:<{COL}} {fmt(b_val):>{W}}   {fmt(i_val):>{W}}")

    def divider():
        print("  " + "-" * (COL + W * 2 + 5))

    print()
    print("=" * 70)
    print("  OLYMPIFLOW — COLISEUM CONGESTION STATS  (for slide S17)")
    print("=" * 70)
    print(f"  Venue    : {data['venue']}")
    print(f"  Capacity : {data['capacity']:,} seats")
    print(f"  Event    : Athletics (18:00 peak arrival)")
    print()
    print(f"  {'Metric':<{COL}} {'Baseline (1-hr peak)':>{W}}   {'Intervention (3-hr)':>{W}}")
    divider()
    row("Scenario", b['label'][:W], i['label'][:W])
    divider()
    row("v/c ratio",
        f"{b['vc_ratio']:.4f}",
        f"{i['vc_ratio']:.4f}")
    row("BPR travel-time multiplier",
        f"{b['bpr_multiplier']:.4f}×",
        f"{i['bpr_multiplier']:.4f}×")
    row("Congestion index (0–100)",
        f"{b['congestion_index']:.1f}",
        f"{i['congestion_index']:.1f}")
    row("Est. travel time (min)",
        f"{b['estimated_travel_time_min']:.1f} min",
        f"{i['estimated_travel_time_min']:.1f} min")
    divider()
    print(f"\n  Travel-time improvement from staggered arrivals: {data['reduction_pct']}%")
    print(f"  Time saved per commuter: "
          f"{b['estimated_travel_time_min'] - i['estimated_travel_time_min']:.1f} min")
    print()
    print("=" * 70)
    print("  USE THESE NUMBERS IN S17:")
    print("=" * 70)
    print(f"  • Baseline v/c ratio              : {b['vc_ratio']}")
    print(f"  • Olympic-day BPR multiplier      : {b['bpr_multiplier']}× (travel time)")
    print(f"  • Congestion index (no action)    : {b['congestion_index']}/100")
    print(f"  • Travel time (no action)         : {b['estimated_travel_time_min']} min "
          f"vs 20-min free-flow (+{b['estimated_travel_time_min'] - 20:.1f} min delay)")
    print(f"  • Staggered arrival v/c           : {i['vc_ratio']}")
    print(f"  • Staggered arrival travel time   : {i['estimated_travel_time_min']} min")
    print(f"  • Travel-time reduction           : {data['reduction_pct']}%")
    print(f"  • Minutes saved per commuter      : "
          f"{b['estimated_travel_time_min'] - i['estimated_travel_time_min']:.1f} min")
    print()


if __name__ == "__main__":
    data = fetch()
    print_table(data)
    print("Raw JSON:")
    print(json.dumps(data, indent=2))

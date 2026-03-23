"""
Generate batch itineraries using GraphHopper for Magdeburg.

Example:
    python generate_itineraries.py --count 150 --output itineraries.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from graphhopper_client import GraphHopperClient
from agent import Agent
from personalised_router import PersonalisedRouter


DEFAULT_START_DATE = "2025-10-10"
DEFAULT_END_DATE = "2026-10-09"
DEFAULT_TZ = "Europe/Berlin"

# Approximate Magdeburg bounding box (lat, lon)
DEFAULT_BBOX = (52.00, 11.45, 52.20, 11.80)

# Common GTFS zip locations in this project.
DEFAULT_GTFS_CANDIDATES = [
    "data/magdeburg_gtfs.zip",
    "data/*.zip",
]

# Predefined points of interest in Magdeburg
PREDEFINED_LOCATIONS = [
    # University
    {"id": "uni_main", "name": "University Main Campus (OVGU)", "lat": 52.1400, "lon": 11.6458},
    {"id": "uni_med", "name": "University Medical Campus (University Hospital)", "lat": 52.1016, "lon": 11.6177},

    # Train stations (Bahnhof)
    {"id": "hbf", "name": "Magdeburg Hauptbahnhof (Main Station)", "lat": 52.1306, "lon": 11.6278},
    {"id": "neustadt", "name": "Magdeburg-Neustadt Station", "lat": 52.1490, "lon": 11.6418},
    {"id": "buckau", "name": "Magdeburg-Buckau Station", "lat": 52.1183, "lon": 11.6413},

    # Shopping areas / stores
    {"id": "city_carre", "name": "City Carré Shopping Center", "lat": 52.1311, "lon": 11.6316},
    {"id": "allee_center", "name": "Allee-Center Shopping Mall", "lat": 52.1305, "lon": 11.6375},
    {"id": "flora_park", "name": "Flora Park Shopping", "lat": 52.1603, "lon": 11.6054},
    {"id": "hasselbachplatz", "name": "Hasselbachplatz Area", "lat": 52.1206, "lon": 11.6274},
    {"id": "boerdepark", "name": "Bördepark Shopping Center", "lat": 52.0834, "lon": 11.5994},
]

# Agent profiles to use
AGENT_FILES = [
    "agents/agent_altruistic.json",
    "agents/agent_biospheric.json",
    "agents/agent_egoistic.json",
    "agents/agent_hedonic.json",
]


@dataclass(frozen=True)
class Stop:
    stop_id: str
    stop_name: str
    lat: float
    lon: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate batch itineraries.")
    parser.add_argument("--count", type=int, default=150, help="Number of origin/destination pairs.")
    parser.add_argument("--output", default="itineraries.csv", help="Output CSV path.")
    parser.add_argument("--host", default="http://localhost:8080", help="GraphHopper base URL.")
    parser.add_argument("--max-walk", type=int, default=500, help="Max walk meters for PT.")
    parser.add_argument("--limit-solutions", type=int, default=3, help="PT alternatives to consider.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Earliest date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Latest date (YYYY-MM-DD).")
    parser.add_argument(
        "--gtfs",
        default="auto",
        help="GTFS zip path. Use 'auto' to search common project locations.",
    )
    parser.add_argument(
        "--bbox",
        default=",".join(str(v) for v in DEFAULT_BBOX),
        help="Bounding box as min_lat,min_lon,max_lat,max_lon.",
    )
    parser.add_argument(
        "--predefined-ratio",
        type=float,
        default=0.2,
        help="Ratio of locations from predefined POIs (0.0-1.0).",
    )
    parser.add_argument(
        "--random-points-ratio",
        type=float,
        default=0.2,
        help="Ratio of locations from random city points (0.0-1.0).",
    )
    parser.add_argument(
        "--num-random-points",
        type=int,
        default=20,
        help="Number of random city points to generate.",
    )
    return parser.parse_args()


def resolve_gtfs_path(raw_gtfs: str) -> Path:
    """Resolve GTFS path from explicit value or common project locations."""
    if raw_gtfs and raw_gtfs.lower() != "auto":
        return Path(raw_gtfs)

    for candidate in DEFAULT_GTFS_CANDIDATES:
        matches = glob.glob(candidate)
        if matches:
            return Path(matches[0])

    searched = ", ".join(DEFAULT_GTFS_CANDIDATES)
    raise FileNotFoundError(
        "GTFS zip not found. Searched: "
        f"{searched}. Provide --gtfs /path/to/your_gtfs.zip"
    )


def load_agents() -> dict[str, Agent]:
    """Load all agent profiles from JSON files."""
    agents = {}
    for agent_file in AGENT_FILES:
        agent_path = Path(agent_file)
        if not agent_path.exists():
            print(f"Warning: Agent file not found: {agent_file}")
            continue
        try:
            with open(agent_path) as f:
                data = json.load(f)
            agent = Agent.from_dict(data, normalise=False)
            agents[agent.id] = agent
            print(f"  Loaded agent: {agent.id}")
        except Exception as e:
            print(f"  Error loading {agent_file}: {e}")
    return agents


def load_gtfs_stops(gtfs_zip: Path) -> list[Stop]:
    if not gtfs_zip.exists():
        raise FileNotFoundError(f"GTFS zip not found: {gtfs_zip}")

    with zipfile.ZipFile(gtfs_zip, "r") as zf:
        with zf.open("stops.txt") as f:
            rows = csv.DictReader(line.decode("utf-8") for line in f)
            stops = []
            for row in rows:
                try:
                    lat = float(row.get("stop_lat") or "")
                    lon = float(row.get("stop_lon") or "")
                except ValueError:
                    continue
                stop_id = (row.get("stop_id") or "").strip()
                stop_name = (row.get("stop_name") or "").strip()
                if stop_id and stop_name:
                    stops.append(Stop(stop_id=stop_id, stop_name=stop_name, lat=lat, lon=lon))
            return stops


def filter_stops_bbox(stops: list[Stop], bbox: tuple[float, float, float, float]) -> list[Stop]:
    min_lat, min_lon, max_lat, max_lon = bbox
    return [
        s
        for s in stops
        if min_lat <= s.lat <= max_lat and min_lon <= s.lon <= max_lon
    ]


def parse_bbox(raw: str) -> tuple[float, float, float, float]:
    try:
        parts = [float(p.strip()) for p in raw.split(",")]
    except ValueError as exc:
        raise ValueError("bbox must be four numbers: min_lat,min_lon,max_lat,max_lon") from exc
    if len(parts) != 4:
        raise ValueError("bbox must be four numbers: min_lat,min_lon,max_lat,max_lon")
    return parts[0], parts[1], parts[2], parts[3]


def random_departure(start_date: str, end_date: str, tz_name: str) -> datetime:
    tz = ZoneInfo(tz_name)
    start = datetime.fromisoformat(start_date).replace(tzinfo=tz)
    end = datetime.fromisoformat(end_date).replace(tzinfo=tz)
    if end <= start:
        raise ValueError("end_date must be after start_date")
    delta_minutes = int((end - start).total_seconds() // 60)
    offset = random.randint(0, max(delta_minutes, 1))
    return start + timedelta(minutes=offset)


def generate_random_city_points(bbox: tuple[float, float, float, float], count: int) -> list[Stop]:
    """Generate random points within the city bounding box."""
    min_lat, min_lon, max_lat, max_lon = bbox
    points = []
    for i in range(count):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        points.append(
            Stop(
                stop_id=f"random_{i+1}",
                stop_name=f"Random City Point {i+1}",
                lat=lat,
                lon=lon,
            )
        )
    return points


def get_predefined_stops() -> list[Stop]:
    """Convert predefined locations to Stop objects."""
    return [
        Stop(
            stop_id=loc["id"],
            stop_name=loc["name"],
            lat=loc["lat"],
            lon=loc["lon"],
        )
        for loc in PREDEFINED_LOCATIONS
    ]


def pick_pairs(
    gtfs_stops: list[Stop],
    predefined_stops: list[Stop],
    random_points: list[Stop],
    count: int,
    predefined_ratio: float = 0.2,
    random_points_ratio: float = 0.2,
) -> list[tuple[Stop, Stop]]:
    """Pick origin-destination pairs from mixed sources.
    
    Args:
        gtfs_stops: List of GTFS stops
        predefined_stops: List of predefined POIs (university, stores, bahnhof)
        random_points: List of random city points
        count: Number of pairs to generate
        predefined_ratio: Probability of selecting from predefined POIs
        random_points_ratio: Probability of selecting from random city points
        
    The remaining probability (1 - predefined_ratio - random_points_ratio) is for GTFS stops.
    """
    if len(gtfs_stops) < 2:
        raise ValueError("Need at least two GTFS stops to sample pairs.")

    gtfs_ratio = 1.0 - predefined_ratio - random_points_ratio
    if not (0.0 <= predefined_ratio <= 1.0 and 0.0 <= random_points_ratio <= 1.0):
        raise ValueError("Ratios must be between 0.0 and 1.0")
    if gtfs_ratio < 0:
        raise ValueError("Sum of predefined_ratio and random_points_ratio cannot exceed 1.0")
    
    def select_location() -> Stop:
        """Select a location based on the configured ratios."""
        rand = random.random()
        if rand < predefined_ratio and predefined_stops:
            return random.choice(predefined_stops)
        elif rand < predefined_ratio + random_points_ratio and random_points:
            return random.choice(random_points)
        else:
            return random.choice(gtfs_stops)
    
    pairs = []
    for _ in range(count):
        origin = select_location()
        dest = select_location()
        
        # Try to ensure origin and destination are different
        max_attempts = 10
        attempts = 0
        while (
            (origin.stop_id == dest.stop_id)
            or (abs(dest.lat - origin.lat) < 0.0001 and abs(dest.lon - origin.lon) < 0.0001)
        ) and attempts < max_attempts:
            dest = select_location()
            attempts += 1
        
        pairs.append((origin, dest))
    
    return pairs


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Load agents
    print("Loading agents...")
    agents = load_agents()
    if not agents:
        raise RuntimeError("No agents loaded. Check agents/ directory.")
    print()

    gtfs_path = resolve_gtfs_path(args.gtfs)
    print(f"Using GTFS: {gtfs_path}")

    stops = load_gtfs_stops(gtfs_path)
    if not stops:
        raise RuntimeError("No GTFS stops loaded.")

    bbox = parse_bbox(args.bbox)
    stops = filter_stops_bbox(stops, bbox)
    if not stops:
        raise RuntimeError("No GTFS stops found inside the bounding box.")

    # Prepare all location sources
    predefined_stops = get_predefined_stops()
    random_points = generate_random_city_points(bbox, args.num_random_points)
    
    print(f"Location sources:")
    print(f"  - GTFS stops: {len(stops)}")
    print(f"  - Predefined POIs: {len(predefined_stops)}")
    print(f"  - Random city points: {len(random_points)}")
    print(f"  - Selection ratios: Predefined={args.predefined_ratio:.1%}, Random={args.random_points_ratio:.1%}, GTFS={1-args.predefined_ratio-args.random_points_ratio:.1%}")
    print()

    # Create PersonalisedRouter with GraphHopper client
    gh_client = GraphHopperClient(base_url=args.host)
    p_router = PersonalisedRouter(gh_client)

    pairs = pick_pairs(
        stops,
        predefined_stops,
        random_points,
        args.count,
        predefined_ratio=args.predefined_ratio,
        random_points_ratio=args.random_points_ratio,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "from_stop_id",
                "from_stop_name",
                "from_lat",
                "from_lon",
                "to_stop_id",
                "to_stop_name",
                "to_lat",
                "to_lon",
                "departure_time",
                "agent_id",
                "agent_profile",
                "rank",
                "mode_key",
                "distance_m",
                "duration_s",
                "utility_score",
                "status",
                "error",
            ],
        )
        writer.writeheader()

        total_rows = 0
        for i, (a, b) in enumerate(pairs, 1):
            departure_dt = random_departure(args.start_date, args.end_date, DEFAULT_TZ)
            departure_iso = departure_dt.isoformat()

            # Route for each agent
            for agent_id, agent in agents.items():
                status = "ok"
                error = ""
                scored_routes = []

                try:
                    scored_routes = p_router.route(
                        agent,
                        a.lat,
                        a.lon,
                        b.lat,
                        b.lon,
                        departure=departure_iso,
                        max_walk_m=args.max_walk,
                    )
                    if not scored_routes:
                        status = "no_route"
                except Exception as exc:
                    status = "error"
                    error = str(exc)

                if status == "ok":
                    # Write each ranked route for this agent
                    for rank, scored_route in enumerate(scored_routes[:3], 1):  # Top 3 routes
                        writer.writerow(
                            {
                                "pair_id": i,
                                "from_stop_id": a.stop_id,
                                "from_stop_name": a.stop_name,
                                "from_lat": f"{a.lat:.6f}",
                                "from_lon": f"{a.lon:.6f}",
                                "to_stop_id": b.stop_id,
                                "to_stop_name": b.stop_name,
                                "to_lat": f"{b.lat:.6f}",
                                "to_lon": f"{b.lon:.6f}",
                                "departure_time": departure_iso,
                                "agent_id": agent_id,
                                "agent_profile": agent.metadata.get("_profile_type", "custom"),
                                "rank": rank,
                                "mode_key": scored_route.mode_key,
                                "distance_m": f"{scored_route.route.total_distance_m:.1f}",
                                "duration_s": f"{scored_route.route.total_duration_s:.1f}",
                                "utility_score": f"{scored_route.utility_score:.1f}",
                                "status": status,
                                "error": error,
                            }
                        )
                        total_rows += 1
                else:
                    # Write error row for this agent
                    writer.writerow(
                        {
                            "pair_id": i,
                            "from_stop_id": a.stop_id,
                            "from_stop_name": a.stop_name,
                            "from_lat": f"{a.lat:.6f}",
                            "from_lon": f"{a.lon:.6f}",
                            "to_stop_id": b.stop_id,
                            "to_stop_name": b.stop_name,
                            "to_lat": f"{b.lat:.6f}",
                            "to_lon": f"{b.lon:.6f}",
                            "departure_time": departure_iso,
                            "agent_id": agent_id,
                            "agent_profile": agent.metadata.get("_profile_type", "custom"),
                            "rank": "",
                            "mode_key": "",
                            "distance_m": "",
                            "duration_s": "",
                            "utility_score": "",
                            "status": status,
                            "error": error,
                        }
                    )
                    total_rows += 1

    print(f"Wrote {total_rows} rows to {output_path}")


if __name__ == "__main__":
    main()

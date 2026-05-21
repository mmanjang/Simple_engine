# Low-level client that talks directly to the GraphHopper HTTP server.


import requests
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional


# Data classes
@dataclass
class WalkLeg:
    type: str = "walk"
    distance_m: float = 0.0
    duration_s: float = 0.0
    geometry: Optional[dict] = None  # GeoJSON LineString
    
    @property
    def mode(self):
        return "walk"


@dataclass
class PtLeg:
    type: str = "pt"
    route_id: str = ""
    trip_headsign: str = ""
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    from_stop: str = ""
    to_stop: str = ""
    num_stops: int = 0
    stops: list = field(default_factory=list)
    geometry: Optional[dict] = None  # GeoJSON LineString
    
    @property
    def mode(self):
        return "pt"
    
    @property
    def distance_m(self):
        """Calculate distance from stops if available"""
        # This would need to be calculated from geometry
        # For now, return 0 or estimate from stop distances and come back to it later
        return 0.0
    
    @property
    def duration_s(self):
        """Calculate duration from departure/arrival times"""
        if self.departure_time and self.arrival_time:
            try:
                from datetime import datetime
                dep = datetime.fromisoformat(self.departure_time.replace('Z', '+00:00'))
                arr = datetime.fromisoformat(self.arrival_time.replace('Z', '+00:00'))
                return (arr - dep).total_seconds()
            except:
                pass
        return 0.0


@dataclass
class Route:
    distance_m: float = 0.0
    duration_s: float = 0.0
    transfers: int = 0
    legs: list = field(default_factory=list)
    points: dict = field(default_factory=dict)

    @property
    def has_pt_legs(self) -> bool:
        """True if this route contains at least one real PT (tram/bus/train) leg."""
        return any(isinstance(l, PtLeg) for l in self.legs)

    @property
    def duration_min(self) -> float:
        return round(self.duration_s / 60, 1)

    @property
    def distance_km(self) -> float:
        return round(self.distance_m / 1000, 2)
    
    @property
    def geometry(self):
        """Return overall route geometry in GeoJSON format"""
        if self.points and 'coordinates' in self.points:
            return {
                "type": "LineString",
                "coordinates": self.points['coordinates']
            }
        return None


#  Client 

class GraphHopperClient:

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url  = base_url.rstrip("/")
        self.route_url = f"{self.base_url}/route"

    def is_alive(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=3)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def route_car(self, from_lat, from_lon, to_lat, to_lon):
        return self._route_standard(from_lat, from_lon, to_lat, to_lon, "car")

    def route_bike(self, from_lat, from_lon, to_lat, to_lon):
        return self._route_standard(from_lat, from_lon, to_lat, to_lon, "bike")

    def route_foot(self, from_lat, from_lon, to_lat, to_lon):
        return self._route_standard(from_lat, from_lon, to_lat, to_lon, "foot")

    def route_pt(self, from_lat, from_lon, to_lat, to_lon,
                 departure_time=None, arrive_by=False,
                 max_walk_meters=1500, limit_solutions=3):

        if departure_time is None:
            departure_time = datetime.now(tz=timezone.utc)
        if departure_time.tzinfo is None:
            departure_time = departure_time.replace(tzinfo=timezone.utc)

        params = {
            "point": [
                f"{from_lat},{from_lon}",
                f"{to_lat},{to_lon}",
            ],
            "profile":                    "pt",
            "pt.earliest_departure_time": departure_time.isoformat(),
            "pt.arrive_by":               str(arrive_by).lower(),
            "pt.max_walk_distance_meter": max_walk_meters,
            "pt.limit_solutions":         limit_solutions,
            "locale":                     "en",
            "points_encoded":             False,
            "details":                    ["time", "distance"],  # Request detailed info
        }
        return self._send(params)

    #  Internal methods

    def _route_standard(self, from_lat, from_lon, to_lat, to_lon, profile):
        params = {
            "point": [
                f"{from_lat},{from_lon}",
                f"{to_lat},{to_lon}",
            ],
            "profile":        profile,
            "locale":         "en",
            "points_encoded": False,
        }
        return self._send(params)

    def _send(self, params: dict) -> list[Route]:
        try:
            resp = requests.get(self.route_url, params=params, timeout=30)
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot reach GraphHopper at {self.base_url}. Is it running?"
            )

        if resp.status_code != 200:
            try:
                detail = resp.json().get("message", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"GraphHopper error {resp.status_code}: {detail}")

        data = resp.json()
        if "paths" not in data or not data["paths"]:
            return []

        return [self._parse_path(p) for p in data["paths"]]

    def _extract_leg_geometry(self, leg_data: dict, overall_points: dict) -> Optional[dict]:

        # 1. Direct geometry from the leg
        geom = leg_data.get("geometry")
        if isinstance(geom, dict) and geom.get("type") == "LineString" and geom.get("coordinates"):
            return geom

        # 2. Some responses may use 'points'
        points = leg_data.get("points")
        if isinstance(points, dict) and points.get("coordinates"):
            return {
                "type": "LineString",
                "coordinates": points["coordinates"]
            }

        # 3. PT fallback: build shape from stop geometries
        if leg_data.get("type") == "pt":
            coords = []
            for stop in leg_data.get("stops", []):
                stop_geom = stop.get("geometry", {})
                if isinstance(stop_geom, dict) and stop_geom.get("coordinates"):
                    lon, lat = stop_geom["coordinates"][:2]
                    coords.append([lon, lat])

            if len(coords) >= 2:
                return {
                    "type": "LineString",
                    "coordinates": coords
                }

        # 4. No reliable geometry found
        return None
    def _parse_path(self, path: dict) -> Route:
        # transfers can be -1 in GH when the journey is walk-only (no PT legs).
        # Clamp to 0 so the display makes sense.
        raw_transfers = path.get("transfers", 0)
        transfers = max(0, raw_transfers) if raw_transfers is not None else 0

        route = Route(
            distance_m = path.get("distance", 0),
            duration_s = path.get("time", 0) / 1000,
            transfers  = transfers,
            points     = path.get("points", {}),
        )

        overall_points = path.get("points", {})
        
        for leg_data in path.get("legs", []):
            leg_type = leg_data.get("type", "")
            leg_geometry = self._extract_leg_geometry(leg_data, overall_points)

            if leg_type == "walk":
                distance_m = leg_data.get("distance", 0)
                duration_s = leg_data.get("time", 0) / 1000 if leg_data.get("time") else 0
                
                # If GraphHopper didn't provide duration, estimate it
                # Walking speed: ~5 km/h = 1.39 m/s
                if duration_s == 0 and distance_m > 0:
                    duration_s = distance_m / 1.39
                
                route.legs.append(WalkLeg(
                    distance_m = distance_m,
                    duration_s = duration_s,
                    geometry   = leg_geometry,
                ))

            elif leg_type == "pt":
                stops     = leg_data.get("stops", [])
                from_stop = stops[0].get("stop_name", "?")  if stops else "?"
                to_stop   = stops[-1].get("stop_name", "?") if stops else "?"
                dep_time  = stops[0].get("departure_time")  if stops else None
                arr_time  = stops[-1].get("arrival_time")   if stops else None

                route.legs.append(PtLeg(
                    route_id       = leg_data.get("route_id", ""),
                    trip_headsign  = leg_data.get("trip_headsign", ""),
                    departure_time = dep_time,
                    arrival_time   = arr_time,
                    from_stop      = from_stop,
                    to_stop        = to_stop,
                    num_stops      = len(stops),
                    stops          = stops,
                    geometry       = leg_geometry,
                ))

        return route
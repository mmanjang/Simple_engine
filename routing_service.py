

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent import Agent
from graphhopper_client import GraphHopperClient
from personalised_router import PersonalisedRouter


Coordinate = Dict[str, float]
CognitivePassport = Dict[str, Any]


class RoutingServiceError(Exception):
    """Raised when the routing service cannot complete a request."""


class RoutingService:
    """Stateless wrapper around GraphHopper + PersonalisedRouter."""

    def __init__(
        self,
        graphhopper_host: str = "http://localhost:8080",
        pois: Optional[List[Dict[str, Any]]] = None,
        poi_proximity_m: int = 100,
        check_graphhopper: bool = True,
    ) -> None:
        self.graphhopper_host = graphhopper_host
        self.client = GraphHopperClient(base_url=graphhopper_host)
        if check_graphhopper and not self.client.is_alive():
            raise RoutingServiceError(
                f"GraphHopper is not reachable at {graphhopper_host}. Start GraphHopper first."
            )
        self.router = PersonalisedRouter(
            self.client,
            pois=pois or [],
            poi_proximity_m=poi_proximity_m,
        )

    def rank_routes(
        self,
        cognitive_passport: CognitivePassport,
        start: Coordinate,
        stop: Coordinate,
        departure: Optional[str] = None,
        max_walk_m: int = 500,
        include_unavailable: bool = True,
    ) -> Dict[str, Any]:
        """Compute and serialize personalised ranked routes."""
        agent = self._agent_from_cognitive_passport(cognitive_passport)
        from_lat, from_lon = self._validate_coordinate(start, "start")
        to_lat, to_lon = self._validate_coordinate(stop, "stop")

        results = self.router.route(
            agent,
            from_lat,
            from_lon,
            to_lat,
            to_lon,
            departure=departure,
            max_walk_m=max_walk_m,
        )

        if not include_unavailable:
            results = [route for route in results if getattr(route, "available", True)]

        routes = [self._scored_route_to_dict(route, agent) for route in results]
        best_route = routes[0] if routes else None

        return {
            "status": "ok" if routes else "no_routes_found",
            "request": {
                "start": {"lat": from_lat, "lon": from_lon},
                "stop": {"lat": to_lat, "lon": to_lon},
                "departure": departure,
                "max_walk_m": max_walk_m,
            },
            "agent": self._agent_summary(agent),
            "straight_line_distance_m": self._haversine_m(from_lat, from_lon, to_lat, to_lon),
            "best_route": best_route,
            "routes": routes,
            "metadata": {
                "route_count": len(routes),
                "graphhopper_host": self.graphhopper_host,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    @staticmethod
    def _agent_from_cognitive_passport(cognitive_passport: CognitivePassport) -> Agent:
        if not isinstance(cognitive_passport, dict):
            raise RoutingServiceError("cognitive_passport must be a JSON object.")

        values = cognitive_passport.get("values", {}) or {}
        if not isinstance(values, dict):
            raise RoutingServiceError("cognitive_passport.values must be a JSON object.")

        all_norm = all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in values.values())
        return Agent.from_dict(cognitive_passport, normalise=not all_norm)

    @staticmethod
    def _validate_coordinate(coord: Coordinate, label: str) -> tuple[float, float]:
        try:
            lat = float(coord["lat"])
            lon = float(coord["lon"])
        except Exception as exc:
            raise RoutingServiceError(f"{label} must contain numeric 'lat' and 'lon'.") from exc

        if not -90 <= lat <= 90:
            raise RoutingServiceError(f"{label}.lat must be between -90 and 90.")
        if not -180 <= lon <= 180:
            raise RoutingServiceError(f"{label}.lon must be between -180 and 180.")
        return lat, lon

    @staticmethod
    def _haversine_m(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> float:
        dlat = math.radians(to_lat - from_lat)
        dlon = math.radians(to_lon - from_lon)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(from_lat))
            * math.cos(math.radians(to_lat))
            * math.sin(dlon / 2) ** 2
        )
        return round(6_371_000 * 2 * math.asin(math.sqrt(a)), 2)

    @staticmethod
    def _agent_summary(agent: Agent) -> Dict[str, Any]:
        return {
            "id": getattr(agent, "id", None),
            "values": dict(getattr(agent, "value_weights", {}) or {}),
            "beliefs": dict(getattr(agent, "beliefs", {}) or {}),
            "available_modes": list(agent.available_modes()) if hasattr(agent, "available_modes") else [],
            "top_values": [
                {"dimension": dim, "weight": weight}
                for dim, weight in (agent.top_values(3) if hasattr(agent, "top_values") else [])
            ],
        }

    @classmethod
    def _scored_route_to_dict(cls, sr: Any, agent: Agent) -> Dict[str, Any]:
        route = getattr(sr, "route", None)
        dimension_scores = [cls._dimension_score_to_dict(ds) for ds in getattr(sr, "dimension_scores", [])]
        top_matching_values = [cls._dimension_score_to_dict(ds) for ds in getattr(sr, "top_matching_values", [])]
        top_conflicting_values = [cls._dimension_score_to_dict(ds) for ds in getattr(sr, "top_conflicting_values", [])]

        return {
            "rank": getattr(sr, "rank", None),
            "mode_key": getattr(sr, "mode_key", None),
            "mode_label": getattr(sr, "mode_label", None),
            "available": getattr(sr, "available", True),
            "availability_reason": cls._availability_reason(sr, agent),
            "score": {
                "utility": getattr(sr, "utility_score", None),
                "raw": getattr(sr, "raw_score", None),
                "poi_boost": getattr(sr, "poi_boost", 0.0),
            },
            "summary": {
                "duration_seconds": getattr(route, "total_duration_s", None),
                "distance_meters": getattr(route, "total_distance_m", None),
                "transfers": getattr(route, "transfers", 0),
            },
            "top_matching_values": top_matching_values,
            "top_conflicting_values": top_conflicting_values,
            "dimension_scores": dimension_scores,
            "matched_pois": list(getattr(sr, "matched_pois", []) or []),
            "legs": [cls._leg_to_dict(leg, index) for index, leg in enumerate(getattr(route, "legs", []) or [], start=1)],
        }

    @staticmethod
    def _dimension_score_to_dict(ds: Any) -> Dict[str, Any]:
        return {
            "dimension": getattr(ds, "dimension", None),
            "agent_weight": getattr(ds, "agent_weight", None),
            "mode_fit": getattr(ds, "blended_attribute", None),
            "contribution": getattr(ds, "contribution", None),
        }

    @staticmethod
    def _leg_to_dict(leg: Any, index: int) -> Dict[str, Any]:
        return {
            "step": index,
            "mode": getattr(leg, "mode", None),
            "from_name": getattr(leg, "from_name", None),
            "to_name": getattr(leg, "to_name", None),
            "route_id": getattr(leg, "route_id", None),
            "distance_meters": getattr(leg, "distance_m", None),
            "duration_seconds": getattr(leg, "duration_s", None),
            "departure_time": getattr(leg, "departure_time", None),
            "arrival_time": getattr(leg, "arrival_time", None),
            "stops": list(getattr(leg, "stops", []) or []),
        }

    @staticmethod
    def _availability_reason(sr: Any, agent: Agent) -> Optional[str]:
        if getattr(sr, "available", True):
            return None

        missing = []
        mode_key = getattr(sr, "mode_key", "") or ""
        beliefs = getattr(agent, "beliefs", {}) or {}
        if "bike" in mode_key and not beliefs.get("owns_bike"):
            missing.append("bike")
        if "car" in mode_key and not beliefs.get("owns_car"):
            missing.append("car")
        if "pt" in mode_key and not beliefs.get("has_pt_access"):
            missing.append("public_transport_access")
        return "missing: " + ", ".join(missing) if missing else "mode unavailable for this agent"

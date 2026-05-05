"""
intermodal_router_v2.py
───────────────────────
Behavior-aware intermodal routing engine.

Purpose
-------
Build realistic multimodal route candidates before the personalised value model
scores them. This version improves the original intermodal router by using both:

  1. Trip distance
  2. Transfer burden

The goal is not only to generate technically feasible routes, but to avoid
behaviorally unrealistic options such as complex PT routes for very short trips.

Architecture
------------
GraphHopperClient
    ↓
IntermodalRouter
    - chooses candidate strategies based on distance
    - builds direct and intermodal routes
    - evaluates transfer realism
    - computes a generalized journey cost
    ↓
PersonalisedRouter
    - scores candidates using psychological value weights

This file is intended to replace your existing intermodal_router.py after testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from dateutil import parser as dp

from graphhopper_client import GraphHopperClient, Route, WalkLeg, PtLeg

if TYPE_CHECKING:
    from agent import Agent
else:
    Agent = Any


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntermodalLeg:
    """One leg of an intermodal journey."""

    mode: str                         # "walk" | "bike" | "car" | "pt"
    description: str                  # human-readable label
    distance_m: float = 0.0
    duration_s: float = 0.0
    route_id: str = ""
    trip_headsign: str = ""
    from_name: str = ""
    to_name: str = ""
    from_stop: str = ""
    to_stop: str = ""
    num_stops: int = 0
    stops: list = field(default_factory=list)
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    geometry: Optional[dict] = None   # GeoJSON lineString if available


@dataclass
class IntermodalRoute:
    """A complete candidate journey made of one or more legs."""

    label: str                        # e.g. "🚴 Bike + 🚌 PT"
    strategy: str                     # e.g. "bike_pt"
    legs: list[IntermodalLeg]
    total_duration_s: float = 0.0
    total_distance_m: float = 0.0
    transfers: int = 0
    feasible: bool = True
    infeasible_reason: str = ""
    geometry: dict = field(default_factory=dict)

    # Added for improved intermodal decision logic
    generalized_cost_s: float = 0.0
    realism_score: float = 1.0
    decision_reason: str = ""

    @property
    def duration_min(self) -> int:
        return int(self.total_duration_s // 60)

    @property
    def distance_km(self) -> float:
        return round(self.total_distance_m / 1000, 2)


@dataclass
class IntermodalPolicy:
    """
    Policy parameters for behavior-aware candidate generation.

    These values are intentionally explicit so they can later be justified,
    tuned, or learned from survey/validation data.
    """

    # Distance regimes in km
    very_short_km: float = 1.5
    short_km: float = 3.0
    medium_km: float = 8.0
    long_km: float = 15.0

    # Practical mode limits in km
    walk_max_km: float = 5.0
    bike_max_km: float = 15.0
    car_pt_min_km: float = 5.0

    # Transfer realism
    transfer_penalty_min: float = 7.0
    pt_boarding_penalty_min: float = 4.0
    first_mile_penalty_min: float = 3.0

    # Strategy-specific transfer tolerance
    max_transfers_very_short: int = 0
    max_transfers_short: int = 1
    max_transfers_medium: int = 2
    max_transfers_long: int = 3

    # First-mile search
    max_candidate_boarding_stops: int = 3
    stop_search_walk_m: int = 2000


# ─────────────────────────────────────────────────────────────────────────────
# Intermodal Router
# ─────────────────────────────────────────────────────────────────────────────

class IntermodalRouter:
    """
    Builds realistic intermodal journey candidates.

    Improvements over the earlier version:
    - Strategy selection depends on distance regime.
    - Transfer tolerance depends on trip distance.
    - Routes are evaluated using generalized journey cost.
    - Infeasible routes can be retained for debugging with explanations.
    - Geometry is preserved at leg level when available.
    """

    def __init__(
        self,
        client: GraphHopperClient,
        departure: Optional[str] = None,
        max_walk_m: int = 1500,
        policy: Optional[IntermodalPolicy] = None,
        keep_infeasible: bool = False,
    ):
        self.client = client
        self.departure = departure
        self.max_walk = max_walk_m
        self.policy = policy or IntermodalPolicy()
        self.keep_infeasible = keep_infeasible

        self._stop_cache: dict = {}
        self._route_cache: dict = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def plan(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        agent: Optional[Agent] = None,
    ) -> list[IntermodalRoute]:
        """
        Return realistic route candidates.

        The method still works without an agent, so your existing
        PersonalisedRouter can call it exactly as before. If an agent is passed,
        basic availability and profile-sensitive feasibility can be applied here.
        """

        distance_km = self._estimate_routable_distance_km(
            from_lat, from_lon, to_lat, to_lon
        )

        strategies = self._choose_strategies(distance_km, agent=agent)

        routes: list[IntermodalRoute] = []
        with ThreadPoolExecutor(max_workers=min(6, max(1, len(strategies)))) as executor:
            future_to_strategy = {
                executor.submit(
                    self._execute,
                    strategy,
                    from_lat,
                    from_lon,
                    to_lat,
                    to_lon,
                    distance_km,
                ): strategy
                for strategy in strategies
            }

            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    route = future.result()
                    if route is not None:
                        routes.append(route)
                except Exception as exc:
                    routes.append(IntermodalRoute(
                        label=strategy,
                        strategy=strategy,
                        legs=[],
                        feasible=False,
                        infeasible_reason=f"Strategy failed: {exc}",
                        decision_reason=f"Strategy failed: {exc}",
                    ))

        evaluated: list[IntermodalRoute] = []
        for route in routes:
            self._evaluate_route_realism(route, distance_km, agent=agent)
            if route.feasible or self.keep_infeasible:
                evaluated.append(route)

        evaluated.sort(key=lambda r: (
            not r.feasible,
            r.generalized_cost_s if r.generalized_cost_s > 0 else r.total_duration_s,
            r.transfers,
            r.total_duration_s,
        ))

        return evaluated

    # ──────────────────────────────────────────────────────────────────────
    # Strategy selection
    # ──────────────────────────────────────────────────────────────────────

    def _max_transfers_for_distance(self, distance_km: float) -> int:
        """
        Distance-aware transfer tolerance.

        Short trips should be simple. Longer trips can justify more transfers.
        """
        p = self.policy
        if distance_km <= p.very_short_km:
            return p.max_transfers_very_short
        if distance_km <= p.short_km:
            return p.max_transfers_short
        if distance_km <= p.medium_km:
            return p.max_transfers_medium
        return p.max_transfers_long

    def _choose_strategies(
        self,
        distance_km: float,
        agent: Optional[Agent] = None,
    ) -> list[str]:
        """
        Choose which strategies to attempt.

        This is still a generator, not the final decision. Bad candidates are
        later rejected or penalized by _evaluate_route_realism().
        """
        p = self.policy
        max_transfers = self._max_transfers_for_distance(distance_km)
        strategies: list[str] = []

        has_bike = self._agent_can(agent, "bike")
        has_car = self._agent_can(agent, "car")
        has_pt = self._agent_can(agent, "pt")

        # Very short trips: simple local mobility only.
        if distance_km <= p.very_short_km:
            strategies.append("foot_direct")
            if has_bike:
                strategies.append("bike_direct")
            if has_car:
                strategies.append("car_direct")
            # PT only if there is a chance of direct/no-transfer service.
            if has_pt and max_transfers == 0:
                strategies.append("pt_direct")
            return self._dedupe(strategies)

        # Short trips: walking/bike/direct PT are plausible; car also included
        # if available because it may be a realistic baseline.
        if distance_km <= p.short_km:
            if distance_km <= p.walk_max_km:
                strategies.append("foot_direct")
            if has_bike:
                strategies.append("bike_direct")
            if has_car:
                strategies.append("car_direct")
            if has_pt:
                strategies.append("pt_direct")
            return self._dedupe(strategies)

        # Medium trips: PT and bike become important. Bike+PT is plausible.
        if distance_km <= p.medium_km:
            if has_pt:
                strategies.append("pt_direct")
            if has_bike and distance_km <= p.bike_max_km:
                strategies.append("bike_direct")
            if has_car:
                strategies.append("car_direct")
            if has_bike and has_pt and max_transfers >= 1:
                strategies.append("bike_pt")
            if has_car and has_pt and distance_km >= p.car_pt_min_km and max_transfers >= 2:
                strategies.append("car_pt")
            return self._dedupe(strategies)

        # Long trips: walking should disappear; bike direct only if still within
        # practical limit. PT and car baselines remain important.
        if has_pt:
            strategies.append("pt_direct")
        if has_car:
            strategies.append("car_direct")
        if has_bike and distance_km <= p.bike_max_km:
            strategies.append("bike_direct")
        if has_bike and has_pt:
            strategies.append("bike_pt")
        if has_car and has_pt:
            strategies.append("car_pt")

        return self._dedupe(strategies)

    # ──────────────────────────────────────────────────────────────────────
    # Strategy execution
    # ──────────────────────────────────────────────────────────────────────

    def _execute(
        self,
        strategy: str,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        distance_km: float,
    ) -> Optional[IntermodalRoute]:
        try:
            if strategy == "pt_direct":
                return self._pt_direct(from_lat, from_lon, to_lat, to_lon)

            if strategy == "car_direct":
                return self._direct(
                    from_lat, from_lon, to_lat, to_lon,
                    "car", "🚗 Drive", "car_direct"
                )

            if strategy == "bike_direct":
                return self._direct(
                    from_lat, from_lon, to_lat, to_lon,
                    "bike", "🚴 Bike", "bike_direct"
                )

            if strategy == "foot_direct":
                return self._direct(
                    from_lat, from_lon, to_lat, to_lon,
                    "foot", "🚶 Walk", "foot_direct"
                )

            if strategy == "bike_pt":
                return self._first_mile(
                    from_lat, from_lon, to_lat, to_lon,
                    "bike", "🚴 Bike + 🚌 PT", "bike_pt"
                )

            if strategy == "car_pt":
                return self._first_mile(
                    from_lat, from_lon, to_lat, to_lon,
                    "car", "🚗 Drive + 🚌 PT", "car_pt"
                )

            return None

        except Exception as exc:
            return IntermodalRoute(
                label=strategy,
                strategy=strategy,
                legs=[],
                feasible=False,
                infeasible_reason=str(exc),
                decision_reason=str(exc),
            )

    def _direct(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        mode: str,
        label: str,
        strategy: str,
    ) -> Optional[IntermodalRoute]:
        """Single-mode route."""
        routes = getattr(self.client, f"route_{mode}")(
            from_lat, from_lon, to_lat, to_lon
        )
        if not routes:
            return None

        r = routes[0]
        leg = IntermodalLeg(
            mode=mode,
            description=label,
            distance_m=r.distance_m,
            duration_s=r.duration_s,
            from_name=f"{from_lat:.4f},{from_lon:.4f}",
            to_name=f"{to_lat:.4f},{to_lon:.4f}",
            geometry=r.geometry,
        )

        return IntermodalRoute(
            label=label,
            strategy=strategy,
            legs=[leg],
            total_duration_s=r.duration_s,
            total_distance_m=r.distance_m,
            transfers=0,
            geometry=r.geometry or r.points or {},
        )

    def _pt_direct(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
    ) -> Optional[IntermodalRoute]:
        """Direct PT journey: access walk + PT + egress walk."""
        dep_dt = dp.parse(self.departure) if self.departure else datetime.now(tz=timezone.utc)

        routes = self.client.route_pt(
            from_lat,
            from_lon,
            to_lat,
            to_lon,
            departure_time=dep_dt,
            max_walk_meters=self.max_walk,
            limit_solutions=3,
        )
        if not routes:
            return None

        # Prefer low generalized burden already at this stage: duration + transfer penalty.
        best = min(
            routes,
            key=lambda r: r.duration_s + max(0, r.transfers) * self.policy.transfer_penalty_min * 60,
        )

        has_pt = any(isinstance(leg, PtLeg) for leg in best.legs)
        if not has_pt:
            return IntermodalRoute(
                label="🚌 Public Transport",
                strategy="pt_direct",
                legs=[],
                feasible=False,
                infeasible_reason="No transit service found for this departure time.",
                decision_reason="PT response contained no real PT leg.",
            )

        legs = self._convert_pt_legs(best)
        legs_distance = sum(leg.distance_m for leg in legs)
        total_distance = legs_distance if legs_distance > best.distance_m * 1.5 else best.distance_m

        return IntermodalRoute(
            label="🚌 Public Transport",
            strategy="pt_direct",
            legs=legs,
            total_duration_s=best.duration_s,
            total_distance_m=total_distance,
            transfers=best.transfers,
            geometry=best.geometry or best.points or {},
        )

    def _first_mile(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        first_mode: str,
        label: str,
        strategy: str,
    ) -> Optional[IntermodalRoute]:
        """
        Bike/car to a candidate PT stop, then PT to destination.

        Compared with the old approach, candidate stop selection is still based
        on PT alternatives, but candidates are ranked by total generalized cost,
        not only raw duration.
        """
        dep_dt = dp.parse(self.departure) if self.departure else datetime.now(tz=timezone.utc)

        boarding_stops = self._find_candidate_boarding_stops(
            from_lat, from_lon, to_lat, to_lon, dep_dt
        )
        if not boarding_stops:
            return None

        best: Optional[IntermodalRoute] = None
        for stop in boarding_stops[: self.policy.max_candidate_boarding_stops]:
            route = self._combine_first_mile(
                from_lat,
                from_lon,
                stop["lat"],
                stop["lon"],
                stop["name"],
                stop.get("departure"),
                to_lat,
                to_lon,
                first_mode,
                label,
                strategy,
                dep_dt,
            )
            if route and route.feasible:
                self._compute_generalized_cost(route)
                if best is None or route.generalized_cost_s < best.generalized_cost_s:
                    best = route

        return best

    def _combine_first_mile(
        self,
        from_lat: float,
        from_lon: float,
        stop_lat: float,
        stop_lon: float,
        stop_name: str,
        stop_departure: Optional[str],
        to_lat: float,
        to_lon: float,
        first_mode: str,
        label: str,
        strategy: str,
        dep_dt: datetime,
    ) -> Optional[IntermodalRoute]:
        """Combine first-mile mode with PT from the chosen boarding stop."""

        first_routes = getattr(self.client, f"route_{first_mode}")(
            from_lat, from_lon, stop_lat, stop_lon
        )
        if not first_routes:
            return None

        first = first_routes[0]

        # Start PT after the first-mile leg plus a small boarding buffer.
        pt_departure = dep_dt
        if first.duration_s > 0:
            from datetime import timedelta
            pt_departure = dep_dt + timedelta(
                seconds=first.duration_s + self.policy.first_mile_penalty_min * 60
            )

        pt_routes = self.client.route_pt(
            stop_lat,
            stop_lon,
            to_lat,
            to_lon,
            departure_time=pt_departure,
            max_walk_meters=self.max_walk,
            limit_solutions=3,
        )
        if not pt_routes:
            return None

        pt_candidates = [r for r in pt_routes if any(isinstance(l, PtLeg) for l in r.legs)]
        if not pt_candidates:
            return None

        pt = min(
            pt_candidates,
            key=lambda r: r.duration_s + max(0, r.transfers) * self.policy.transfer_penalty_min * 60,
        )

        icon = "🚴" if first_mode == "bike" else "🚗"
        first_leg = IntermodalLeg(
            mode=first_mode,
            description=f"{icon} {first_mode.title()} to {stop_name}",
            distance_m=first.distance_m,
            duration_s=first.duration_s,
            from_name="Origin",
            to_name=stop_name,
            geometry=first.geometry,
        )

        pt_legs = self._convert_pt_legs(pt)
        legs = [first_leg] + pt_legs

        total_duration = first.duration_s + pt.duration_s + self.policy.first_mile_penalty_min * 60
        total_distance = first.distance_m + sum(leg.distance_m for leg in pt_legs)

        return IntermodalRoute(
            label=label,
            strategy=strategy,
            legs=legs,
            total_duration_s=total_duration,
            total_distance_m=total_distance,
            transfers=pt.transfers,
            geometry=self._merge_leg_geometries(legs) or pt.geometry or pt.points or {},
        )

    # ──────────────────────────────────────────────────────────────────────
    # Route realism and generalized cost
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate_route_realism(
        self,
        route: IntermodalRoute,
        distance_km: float,
        agent: Optional[Agent] = None,
    ) -> None:
        """Update route.feasible, generalized cost, and decision reason."""
        if not route.feasible:
            if not route.decision_reason:
                route.decision_reason = route.infeasible_reason
            return

        max_transfers = self._max_transfers_for_distance(distance_km)

        if route.transfers > max_transfers:
            route.feasible = False
            route.infeasible_reason = (
                f"Too many transfers for {distance_km:.1f} km trip: "
                f"{route.transfers} > allowed {max_transfers}."
            )
            route.decision_reason = route.infeasible_reason
            self._compute_generalized_cost(route)
            return

        # Additional practical feasibility by profile, if an agent is provided.
        if agent is not None:
            ok, reason = self.check_feasibility(route, agent)
            if not ok:
                route.feasible = False
                route.infeasible_reason = reason
                route.decision_reason = reason
                self._compute_generalized_cost(route)
                return

        self._compute_generalized_cost(route)
        route.realism_score = self._compute_realism_score(route, distance_km)
        route.decision_reason = self._build_decision_reason(route, distance_km, max_transfers)

    def _compute_generalized_cost(self, route: IntermodalRoute) -> None:
        """
        Generalized journey cost.

        This is not the final personalised utility. It is a behavioral burden
        score used to sort and filter route candidates before value scoring.
        """
        transfer_penalty_s = max(0, route.transfers) * self.policy.transfer_penalty_min * 60
        boarding_penalty_s = 0
        if any(leg.mode == "pt" for leg in route.legs):
            boarding_penalty_s += self.policy.pt_boarding_penalty_min * 60
        if route.strategy in ("bike_pt", "car_pt"):
            boarding_penalty_s += self.policy.first_mile_penalty_min * 60

        route.generalized_cost_s = route.total_duration_s + transfer_penalty_s + boarding_penalty_s

    def _compute_realism_score(self, route: IntermodalRoute, distance_km: float) -> float:
        """0–1 score for how behaviorally plausible the route is."""
        score = 1.0

        # Transfer burden
        score -= min(0.45, 0.15 * max(0, route.transfers))

        # Very short trips should not be overly complex.
        if distance_km <= self.policy.very_short_km and route.strategy in ("pt_direct", "bike_pt", "car_pt"):
            score -= 0.4

        # Penalise long access/egress walking.
        walking_m = sum(leg.distance_m for leg in route.legs if leg.mode == "walk")
        if walking_m > self.max_walk * 2:
            score -= 0.2

        # Intermodal routes need to justify their complexity.
        if route.strategy in ("bike_pt", "car_pt") and route.transfers >= 2:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _build_decision_reason(
        self,
        route: IntermodalRoute,
        distance_km: float,
        max_transfers: int,
    ) -> str:
        parts = [
            f"distance={distance_km:.1f}km",
            f"transfers={route.transfers}/{max_transfers}",
            f"duration={route.duration_min}min",
            f"generalized_cost={int(route.generalized_cost_s // 60)}min",
        ]
        if route.strategy in ("bike_pt", "car_pt"):
            parts.append("intermodal option accepted")
        elif route.strategy == "pt_direct":
            parts.append("direct PT candidate accepted")
        else:
            parts.append("direct mode candidate accepted")
        return " | ".join(parts)

    # ──────────────────────────────────────────────────────────────────────
    # Feasibility helpers
    # ──────────────────────────────────────────────────────────────────────

    def check_feasibility(self, route: IntermodalRoute, agent: Agent) -> tuple[bool, str]:
        """
        Profile-sensitive feasibility checks.

        This keeps your existing PersonalisedRouter compatible: it already calls
        check_feasibility(route, agent). The thresholds here are intentionally
        practical rather than final scientific claims.
        """
        profile_type = "default"
        if hasattr(agent, "infer_profile_type"):
            try:
                profile_type = agent.infer_profile_type()
            except Exception:
                profile_type = "default"

        limits = self._profile_limits(profile_type)
        distance_km = route.total_distance_m / 1000

        if route.strategy == "foot_direct" and distance_km > limits["walk"]:
            return False, (
                f"Walking {distance_km:.1f}km exceeds practical limit "
                f"({limits['walk']}km for {profile_type} profile)."
            )

        if route.strategy == "bike_direct" and distance_km > limits["bike"]:
            return False, (
                f"Cycling {distance_km:.1f}km exceeds practical limit "
                f"({limits['bike']}km for {profile_type} profile)."
            )

        if route.strategy == "bike_pt":
            bike_km = sum(leg.distance_m for leg in route.legs if leg.mode == "bike") / 1000
            if bike_km > limits["bike"]:
                return False, (
                    f"Bike access leg {bike_km:.1f}km exceeds practical limit "
                    f"({limits['bike']}km for {profile_type} profile)."
                )

        return True, ""

    @staticmethod
    def _profile_limits(profile_type: str) -> dict[str, float]:
        """Return walk/bike practical limits by profile."""
        defaults = {"walk": 4.0, "bike": 12.0}
        table = {
            "biospheric": {"walk": 5.0, "bike": 15.0},
            "altruistic": {"walk": 4.0, "bike": 12.0},
            "egoistic": {"walk": 2.5, "bike": 10.0},
            "hedonic": {"walk": 2.0, "bike": 8.0},
        }
        return table.get(profile_type, defaults)

    # ──────────────────────────────────────────────────────────────────────
    # PT conversion and geometry
    # ──────────────────────────────────────────────────────────────────────

    def _convert_pt_legs(self, route: Route) -> list[IntermodalLeg]:
        """Convert GraphHopper Route legs into IntermodalLeg objects."""
        result: list[IntermodalLeg] = []

        for gh_leg in route.legs:
            if isinstance(gh_leg, WalkLeg):
                result.append(IntermodalLeg(
                    mode="walk",
                    description="🚶 Walk",
                    distance_m=gh_leg.distance_m,
                    duration_s=gh_leg.duration_s,
                    geometry=gh_leg.geometry,
                ))

            elif isinstance(gh_leg, PtLeg):
                duration_s = gh_leg.duration_s
                distance_m = self._distance_from_geometry_or_stops(gh_leg.geometry, gh_leg.stops)

                result.append(IntermodalLeg(
                    mode="pt",
                    description=f"🚌 {gh_leg.route_id}" if gh_leg.route_id else "🚌 PT",
                    distance_m=distance_m,
                    duration_s=duration_s,
                    route_id=gh_leg.route_id,
                    trip_headsign=gh_leg.trip_headsign,
                    from_name=gh_leg.from_stop,
                    to_name=gh_leg.to_stop,
                    from_stop=gh_leg.from_stop,
                    to_stop=gh_leg.to_stop,
                    num_stops=gh_leg.num_stops,
                    stops=gh_leg.stops,
                    departure_time=gh_leg.departure_time,
                    arrival_time=gh_leg.arrival_time,
                    geometry=gh_leg.geometry,
                ))

        return result

    def _distance_from_geometry_or_stops(self, geometry: Optional[dict], stops: list) -> float:
        """Distance in metres from GeoJSON geometry, falling back to stop coordinates."""
        coords = []
        if geometry and isinstance(geometry, dict) and geometry.get("coordinates"):
            coords = geometry["coordinates"]
        elif stops:
            for stop in stops:
                stop_geom = stop.get("geometry", {})
                if stop_geom and stop_geom.get("coordinates"):
                    coords.append(stop_geom["coordinates"])

        if len(coords) < 2:
            return 0.0

        distance_m = 0.0
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i][:2]
            lon2, lat2 = coords[i + 1][:2]
            distance_m += self._haversine_km(lat1, lon1, lat2, lon2) * 1000
        return distance_m

    def _merge_leg_geometries(self, legs: list[IntermodalLeg]) -> Optional[dict]:
        """Merge available leg geometries into one GeoJSON LineString."""
        merged = []
        for leg in legs:
            geom = leg.geometry
            if not geom or not isinstance(geom, dict):
                continue
            coords = geom.get("coordinates")
            if not coords:
                continue
            if merged and coords and merged[-1] == coords[0]:
                merged.extend(coords[1:])
            else:
                merged.extend(coords)

        if len(merged) >= 2:
            return {"type": "LineString", "coordinates": merged}
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Stop finding
    # ──────────────────────────────────────────────────────────────────────

    def _find_candidate_boarding_stops(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        dep_dt: datetime,
    ) -> list[dict]:
        """
        Find candidate PT boarding stops by asking GraphHopper for PT routes.

        This is a practical way to discover routable GTFS stops without building
        a separate spatial stop index yet.
        """
        cache_key = (
            round(from_lat, 3), round(from_lon, 3),
            round(to_lat, 3), round(to_lon, 3),
            dep_dt.date().isoformat(),
        )
        if cache_key in self._stop_cache:
            return self._stop_cache[cache_key]

        try:
            pt_routes = self.client.route_pt(
                from_lat,
                from_lon,
                to_lat,
                to_lon,
                departure_time=dep_dt,
                max_walk_meters=self.policy.stop_search_walk_m,
                limit_solutions=3,
            )
        except Exception:
            self._stop_cache[cache_key] = []
            return []

        stops = []
        seen = set()
        for pt_route in pt_routes:
            for leg in pt_route.legs:
                if isinstance(leg, PtLeg) and leg.stops:
                    stop = leg.stops[0]
                    lon, lat = self._stop_lon_lat(stop)
                    name = stop.get("stop_name", "PT Stop")
                    if lat is None or lon is None:
                        continue
                    key = (round(lat, 5), round(lon, 5), name)
                    if key in seen:
                        continue
                    seen.add(key)
                    stops.append({
                        "lat": float(lat),
                        "lon": float(lon),
                        "name": name,
                        "departure": stop.get("departure_time"),
                    })
                    break

        self._stop_cache[cache_key] = stops
        return stops

    @staticmethod
    def _stop_lon_lat(stop: dict) -> tuple[Optional[float], Optional[float]]:
        geom = stop.get("geometry", {})
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        if coords and len(coords) >= 2:
            return coords[0], coords[1]
        lon = stop.get("lon", stop.get("stop_lon"))
        lat = stop.get("lat", stop.get("stop_lat"))
        return lon, lat

    # ──────────────────────────────────────────────────────────────────────
    # Distance and misc helpers
    # ──────────────────────────────────────────────────────────────────────

    def _estimate_routable_distance_km(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
    ) -> float:
        """
        Prefer walking route distance because it accounts for barriers.
        Fall back to haversine if GraphHopper fails.
        """
        crow_km = self._haversine_km(from_lat, from_lon, to_lat, to_lon)
        try:
            walk_routes = self.client.route_foot(from_lat, from_lon, to_lat, to_lon)
            if walk_routes:
                return walk_routes[0].distance_m / 1000
        except Exception:
            pass
        return crow_km

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        import math
        radius_km = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius_km * c

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen = set()
        out = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @staticmethod
    def _agent_can(agent: Optional[Agent], mode: str) -> bool:
        """
        Basic availability check.

        If no agent is provided, assume all modes are technically available so
        candidate generation remains broad. The PersonalisedRouter can still
        mark unavailable routes later.
        """
        if agent is None:
            return True

        if mode == "foot":
            return True
        if mode == "bike":
            return bool(agent.beliefs.get("owns_bike", False))
        if mode == "car":
            return bool(agent.beliefs.get("owns_car", False))
        if mode == "pt":
            return bool(agent.beliefs.get("has_pt_access", False))
        return True

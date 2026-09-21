"""
intermodal_router.py
────────────────────
Behaviour-aware intermodal route candidate generator.

Key design principles
─────────────────────
1. First-mile gate
   A bike or car first-mile leg is only generated if it provides
   *genuine advantage* over walking the same leg:
     - minimum physical distance (bike ≥ 800 m, car ≥ 2 000 m)
     - minimum time saving vs walking the same leg (≥ 4 min)
   Without this gate, the engine produces absurd options like
   "bike 280 m then board tram" — barely better than walking to the stop.

2. Transfer elimination
   For every PT route with ≥ 1 transfer, the engine also evaluates
   whether bike or car can reach the *post-transfer boarding stop*
   directly — skipping the first PT segment and its transfer entirely.
   Example: instead of Tram A → transfer → Tram B, the agent bikes
   1.2 km to the Tram B stop and boards there directly.

3. Egress substitution
   When the PT egress walk is long (≥ egress_min_m), a bike leg from
   the final PT stop to the destination is evaluated as an alternative.

4. Need-vector modulation
   Three places where agent needs affect intermodal logic:
     a. Strategy selection: high comfort/speed suppress foot_direct for
        long legs; high physical boosts active modes.
     b. Transfer tolerance: comfort + reliable weights reduce the number
        of accepted transfers.
     c. Time overhead tolerance: how many extra minutes an intermodal
        option is allowed vs direct PT — comfort and physical relax this,
        speed tightens it.

5. Boarding penalty fix
   The first_mile_penalty is applied exactly once — inside
   _combine_first_mile() when computing pt_departure. It is NOT
   re-added in _compute_generalized_cost().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    mode: str                           # walk | bike | car | pt | wait
    description: str
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
    geometry: Optional[dict] = None


@dataclass
class IntermodalRoute:
    """A complete candidate journey made of one or more legs."""
    label: str
    strategy: str
    legs: list[IntermodalLeg]
    total_duration_s: float = 0.0
    total_distance_m: float = 0.0
    transfers: int = 0
    feasible: bool = True
    infeasible_reason: str = ""
    geometry: dict = field(default_factory=dict)
    generalized_cost_s: float = 0.0
    realism_score: float = 1.0
    decision_reason: str = ""

    # Intermodal metadata
    transfers_eliminated: int = 0       # how many transfers this route saves vs pt_direct
    first_mile_saving_s: float = 0.0    # time saved on first leg vs walking

    @property
    def duration_min(self) -> int:
        return int(self.total_duration_s // 60)

    @property
    def distance_km(self) -> float:
        return round(self.total_distance_m / 1000, 2)


@dataclass
class BoardingStop:
    """A candidate PT boarding stop with metadata."""
    lat: float
    lon: float
    name: str
    departure: Optional[str]
    transfer_index: int = 0     # 0 = first boarding stop, 1 = post-first-transfer, etc.
    walk_time_s: float = 0.0    # estimated walk time from origin to this stop
    pt_leg_index: int = 0       # which PT leg this stop starts


@dataclass
class IntermodalPolicy:
    """
    Policy parameters for behaviour-aware candidate generation.

    Distance regime boundaries and mode limits control which strategies
    are attempted. Need-vector modulation parameters control how agent
    weights shift these boundaries at scoring time.
    """

    # Distance regimes (km)
    very_short_km: float = 1.5
    short_km: float = 3.0
    medium_km: float = 8.0
    long_km: float = 15.0

    # Practical mode limits
    walk_max_km: float = 5.0
    bike_max_km: float = 20.0
    car_pt_min_km: float = 3.0      # lowered: urban car+PT can make sense from 3 km

    # First-mile gate — minimum values below which bike/car first-mile is not worthwhile
    bike_first_mile_min_m: float = 800.0    # bike leg must be at least this long
    car_first_mile_min_m: float = 2000.0    # car leg must be at least this long
    first_mile_min_saving_s: float = 240.0  # must save ≥ 4 min vs walking the same leg

    # Egress substitution
    egress_min_m: float = 600.0     # egress walk must be ≥ this to try bike substitution
    bike_egress_max_m: float = 3000.0

    # Transfer realism
    transfer_penalty_min: float = 7.0       # minutes of generalised cost per transfer
    pt_boarding_penalty_min: float = 4.0    # one-off penalty for boarding PT at all
    first_mile_penalty_min: float = 6.0     # overhead for bike/car first-mile leg
                                            # (parking/securing + walk to platform)
                                            # Applied ONCE in _combine_first_mile only.

    car_overhead_min: float = 8.0           # parking search + walk to/from car

    # Transfer tolerance by distance (base values, modulated by need weights below)
    max_transfers_very_short: int = 0
    max_transfers_short: int = 1
    max_transfers_medium: int = 2
    max_transfers_long: int = 3

    # Need-weight modulation of transfer tolerance
    # comfort/reliable above this threshold reduce max_transfers by 1
    comfort_transfer_threshold: float = 0.6
    reliable_transfer_threshold: float = 0.6

    # Time overhead tolerance: how many extra seconds an intermodal route
    # is allowed vs the best direct PT route. Modulated by need weights.
    base_time_overhead_s: float = 300.0     # 5 min base
    comfort_overhead_bonus_s: float = 240.0 # +4 min if comfort weight is high
    physical_overhead_bonus_s: float = 180.0 # +3 min if physical weight is high
    speed_overhead_penalty_s: float = 300.0 # -5 min if speed weight is high

    # Stop search
    max_candidate_boarding_stops: int = 4
    stop_search_walk_m: int = 2000

    # Walking speed for first-mile time-saving calculation (m/s)
    walk_speed_ms: float = 1.33
    bike_speed_ms: float = 4.17    # ~15 km/h average urban cycling


# ─────────────────────────────────────────────────────────────────────────────
# Intermodal Router
# ─────────────────────────────────────────────────────────────────────────────

class IntermodalRouter:

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
        Return realistic intermodal route candidates for an origin-destination pair.

        Works without an agent (all modes assumed available). When an agent is
        supplied, belief-gating and need-vector modulation are applied.
        """
        distance_km = self._estimate_routable_distance_km(
            from_lat, from_lon, to_lat, to_lon
        )

        strategies = self._choose_strategies(distance_km, agent=agent)

        # Fetch the direct PT route once upfront — we need it for:
        #   a) transfer elimination stop discovery
        #   b) time overhead tolerance baseline
        dep_dt = dp.parse(self.departure) if self.departure else datetime.now(tz=timezone.utc)
        direct_pt_routes = self._cached_pt_routes(
            from_lat, from_lon, to_lat, to_lon, dep_dt
        )
        best_direct_pt_s = self._best_pt_duration(direct_pt_routes)

        # Build execution tasks: (strategy, kwargs)
        tasks = self._build_tasks(
            strategies,
            from_lat, from_lon, to_lat, to_lon,
            distance_km, dep_dt,
            direct_pt_routes, agent,
        )

        routes: list[IntermodalRoute] = []
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(tasks)))) as executor:
            future_map = {
                executor.submit(task_fn, *task_args): task_label
                for task_label, task_fn, task_args in tasks
            }
            for future in as_completed(future_map):
                label = future_map[future]
                try:
                    result = future.result()
                    if result is not None:
                        if isinstance(result, list):
                            routes.extend(result)
                        else:
                            routes.append(result)
                except Exception as exc:
                    routes.append(IntermodalRoute(
                        label=label, strategy=label, legs=[],
                        feasible=False,
                        infeasible_reason=f"Strategy failed: {exc}",
                        decision_reason=f"Strategy failed: {exc}",
                    ))

        # Evaluate realism and apply need-aware overhead tolerance filter
        time_overhead_s = self._compute_time_overhead_tolerance(agent)
        evaluated: list[IntermodalRoute] = []
        for route in routes:
            self._evaluate_route_realism(route, distance_km, agent=agent)

            # Filter intermodal options that are much slower than direct PT
            if (
                route.feasible
                and route.strategy in ("bike_pt", "car_pt", "bike_egress")
                and best_direct_pt_s is not None
                and route.total_duration_s > best_direct_pt_s + time_overhead_s
            ):
                route.feasible = False
                route.infeasible_reason = (
                    f"Intermodal total {route.duration_min} min exceeds "
                    f"direct PT {int(best_direct_pt_s//60)} min "
                    f"+ {int(time_overhead_s//60)} min tolerance."
                )

            if route.feasible or self.keep_infeasible:
                evaluated.append(route)

        # Deduplicate: if transfer-elimination bike_pt and regular bike_pt both
        # found feasible routes, keep only the better one
        evaluated = self._deduplicate_strategies(evaluated)

        evaluated.sort(key=lambda r: (
            not r.feasible,
            r.generalized_cost_s if r.generalized_cost_s > 0 else r.total_duration_s,
            r.transfers,
            r.total_duration_s,
        ))

        return evaluated

    # ──────────────────────────────────────────────────────────────────────
    # Task building
    # ──────────────────────────────────────────────────────────────────────

    def _build_tasks(
        self,
        strategies: list[str],
        from_lat: float, from_lon: float,
        to_lat: float, to_lon: float,
        distance_km: float,
        dep_dt: datetime,
        direct_pt_routes: list,
        agent: Optional[Agent],
    ) -> list[tuple]:
        """
        Build (label, fn, args) tuples for parallel execution.

        For intermodal strategies this also generates transfer-elimination
        variants using post-transfer boarding stops from the direct PT routes.
        """
        p = self.policy
        tasks = []

        # Collect all candidate boarding stops (first boarding + transfer points)
        boarding_stops = self._find_all_boarding_stops(
            from_lat, from_lon, to_lat, to_lon,
            dep_dt, direct_pt_routes
        )

        for strategy in strategies:

            if strategy == "pt_direct":
                tasks.append((
                    "pt_direct",
                    self._pt_direct,
                    (from_lat, from_lon, to_lat, to_lon, direct_pt_routes),
                ))

            elif strategy == "foot_direct":
                tasks.append((
                    "foot_direct",
                    self._direct,
                    (from_lat, from_lon, to_lat, to_lon, "foot", "🚶 Walk", "foot_direct"),
                ))

            elif strategy == "bike_direct":
                tasks.append((
                    "bike_direct",
                    self._direct,
                    (from_lat, from_lon, to_lat, to_lon, "bike", "🚴 Bike", "bike_direct"),
                ))

            elif strategy == "car_direct":
                tasks.append((
                    "car_direct",
                    self._direct,
                    (from_lat, from_lon, to_lat, to_lon, "car", "🚗 Drive", "car_direct"),
                ))

            elif strategy == "bike_pt":
                # Standard first-mile: try all first-boarding stops
                first_mile_stops = [s for s in boarding_stops if s.transfer_index == 0]
                for stop in first_mile_stops[:p.max_candidate_boarding_stops]:
                    tasks.append((
                        "bike_pt",
                        self._first_mile_gated,
                        (from_lat, from_lon, stop, to_lat, to_lon,
                         "bike", "🚴 Bike + 🚌 PT", "bike_pt", dep_dt, False),
                    ))
                # Transfer elimination: try post-transfer stops
                elim_stops = [s for s in boarding_stops if s.transfer_index >= 1]
                for stop in elim_stops[:2]:
                    tasks.append((
                        "bike_pt_elim",
                        self._first_mile_gated,
                        (from_lat, from_lon, stop, to_lat, to_lon,
                         "bike", "🚴 Bike + 🚌 PT (transfer-free)", "bike_pt_elim",
                         dep_dt, True),
                    ))

            elif strategy == "car_pt":
                first_mile_stops = [s for s in boarding_stops if s.transfer_index == 0]
                for stop in first_mile_stops[:p.max_candidate_boarding_stops]:
                    tasks.append((
                        "car_pt",
                        self._first_mile_gated,
                        (from_lat, from_lon, stop, to_lat, to_lon,
                         "car", "🚗 Drive + 🚌 PT", "car_pt", dep_dt, False),
                    ))
                elim_stops = [s for s in boarding_stops if s.transfer_index >= 1]
                for stop in elim_stops[:2]:
                    tasks.append((
                        "car_pt_elim",
                        self._first_mile_gated,
                        (from_lat, from_lon, stop, to_lat, to_lon,
                         "car", "🚗 Drive + 🚌 PT (transfer-free)", "car_pt_elim",
                         dep_dt, True),
                    ))

            elif strategy == "bike_egress":
                tasks.append((
                    "bike_egress",
                    self._egress_substitution,
                    (from_lat, from_lon, to_lat, to_lon,
                     "bike", "🚌 PT + 🚴 Bike Egress", "bike_egress",
                     dep_dt, direct_pt_routes),
                ))

        return tasks

    # ──────────────────────────────────────────────────────────────────────
    # Strategy selection (need-vector aware)
    # ──────────────────────────────────────────────────────────────────────

    def _choose_strategies(
        self,
        distance_km: float,
        agent: Optional[Agent] = None,
    ) -> list[str]:
        """
        Select strategies to attempt based on distance regime, agent beliefs,
        and agent need weights.

        Need-vector modulation:
          - high speed weight  → suppress foot_direct for trips > 1 km
          - high comfort weight → suppress bike_direct for long trips
          - high physical weight → boost active modes, relax bike threshold
          - low speed + high physical → include bike even on medium trips
        """
        p = self.policy
        strategies: list[str] = []

        has_bike = self._agent_can(agent, "bike")
        has_car  = self._agent_can(agent, "car")
        has_pt   = self._agent_can(agent, "pt")

        # Need weights (default 0.5 if no agent)
        speed_w   = self._need(agent, "speed")
        comfort_w = self._need(agent, "comfort")
        physical_w= self._need(agent, "physical")

        # Suppress foot_direct for trips where walking is impractical given agent needs
        # High speed weight → don't offer walking beyond very short trips
        foot_max_km = p.walk_max_km
        if speed_w > 0.7:
            foot_max_km = min(foot_max_km, 1.5)
        elif speed_w > 0.5:
            foot_max_km = min(foot_max_km, 2.5)

        # High physical weight → extend bike range
        bike_max_km = p.bike_max_km
        if physical_w > 0.7:
            bike_max_km = min(bike_max_km * 1.3, 25.0)
        elif comfort_w > 0.7:
            bike_max_km = min(bike_max_km, 12.0)  # comfort-focused: don't push long bike

        # Very short trips
        if distance_km <= p.very_short_km:
            if distance_km <= foot_max_km:
                strategies.append("foot_direct")
            if has_bike and physical_w > 0.4:  # only if agent has some physical interest
                strategies.append("bike_direct")
            if has_car:
                strategies.append("car_direct")
            if has_pt:
                strategies.append("pt_direct")
            return self._dedupe(strategies)

        # Short trips
        if distance_km <= p.short_km:
            if distance_km <= foot_max_km:
                strategies.append("foot_direct")
            if has_bike and distance_km <= bike_max_km:
                strategies.append("bike_direct")
            if has_car:
                strategies.append("car_direct")
            if has_pt:
                strategies.append("pt_direct")
            # Bike+PT only worthwhile if bike leg would be ≥ min threshold
            # For short trips the stop is likely close — gate on physical interest
            if has_bike and has_pt and physical_w > 0.3:
                strategies.append("bike_pt")
            return self._dedupe(strategies)

        # Medium trips
        if distance_km <= p.medium_km:
            if has_pt:
                strategies.append("pt_direct")
            if has_bike and distance_km <= bike_max_km:
                strategies.append("bike_direct")
            if has_car:
                strategies.append("car_direct")
            if has_bike and has_pt:
                strategies.append("bike_pt")
            if has_car and has_pt and distance_km >= p.car_pt_min_km:
                strategies.append("car_pt")
            # Egress substitution: long PT egress common at medium distances
            if has_bike and has_pt:
                strategies.append("bike_egress")
            return self._dedupe(strategies)

        # Long trips
        if has_pt:
            strategies.append("pt_direct")
        if has_car:
            strategies.append("car_direct")
        if has_bike and distance_km <= bike_max_km:
            strategies.append("bike_direct")
        if has_bike and has_pt:
            strategies.append("bike_pt")
        if has_car and has_pt:
            strategies.append("car_pt")
        if has_bike and has_pt:
            strategies.append("bike_egress")

        return self._dedupe(strategies)

    def _max_transfers_for_distance(
        self, distance_km: float, agent: Optional[Agent] = None
    ) -> int:
        """
        Base transfer tolerance by distance, reduced by comfort and reliable weights.
        """
        p = self.policy
        if distance_km <= p.very_short_km:
            base = p.max_transfers_very_short
        elif distance_km <= p.short_km:
            base = p.max_transfers_short
        elif distance_km <= p.medium_km:
            base = p.max_transfers_medium
        else:
            base = p.max_transfers_long

        # Need modulation: comfort + reliable above threshold → 1 fewer transfer
        comfort_w  = self._need(agent, "comfort")
        reliable_w = self._need(agent, "reliable")
        reduction = 0
        if comfort_w >= p.comfort_transfer_threshold:
            reduction += 1
        if reliable_w >= p.reliable_transfer_threshold:
            reduction += 1

        return max(0, base - reduction)

    def _compute_time_overhead_tolerance(
        self, agent: Optional[Agent] = None
    ) -> float:
        """
        How many extra seconds an intermodal route is allowed vs direct PT.

        High comfort or physical weight → more tolerance (worth it for the experience).
        High speed weight → less tolerance (must be competitive).
        """
        p = self.policy
        base = p.base_time_overhead_s
        comfort_w  = self._need(agent, "comfort")
        physical_w = self._need(agent, "physical")
        speed_w    = self._need(agent, "speed")

        overhead = (
            base
            + comfort_w  * p.comfort_overhead_bonus_s
            + physical_w * p.physical_overhead_bonus_s
            - speed_w    * p.speed_overhead_penalty_s
        )
        return max(0.0, overhead)

    # ──────────────────────────────────────────────────────────────────────
    # Strategy execution
    # ──────────────────────────────────────────────────────────────────────

    def _direct(
        self,
        from_lat: float, from_lon: float,
        to_lat: float, to_lon: float,
        mode: str, label: str, strategy: str,
    ) -> Optional[IntermodalRoute]:
        """Single-mode direct route."""
        routes = getattr(self.client, f"route_{mode}")(
            from_lat, from_lon, to_lat, to_lon
        )
        if not routes:
            return None
        r = routes[0]
        leg = IntermodalLeg(
            mode=mode, description=label,
            distance_m=r.distance_m, duration_s=r.duration_s,
            from_name=f"{from_lat:.4f},{from_lon:.4f}",
            to_name=f"{to_lat:.4f},{to_lon:.4f}",
            geometry=r.geometry,
        )
        overhead_s = self.policy.car_overhead_min * 60 if mode == "car" else 0
        return IntermodalRoute(
            label=label, strategy=strategy, legs=[leg],
            total_duration_s=r.duration_s + overhead_s,
            total_distance_m=r.distance_m,
            transfers=0,
            geometry=r.geometry or r.points or {},
        )

    def _pt_direct(
        self,
        from_lat: float, from_lon: float,
        to_lat: float, to_lon: float,
        direct_pt_routes: list,
    ) -> Optional[IntermodalRoute]:
        """Direct PT journey using pre-fetched routes."""
        routes = direct_pt_routes
        if not routes:
            return None

        best = min(
            routes,
            key=lambda r: r.duration_s + max(0, r.transfers) * self.policy.transfer_penalty_min * 60,
        )

        has_pt = any(isinstance(leg, PtLeg) for leg in best.legs)
        if not has_pt:
            return IntermodalRoute(
                label="🚌 Public Transport", strategy="pt_direct",
                legs=[], feasible=False,
                infeasible_reason="No transit service found for this departure time.",
                decision_reason="PT response contained no real PT leg.",
            )

        legs = self._convert_pt_legs(best)
        legs = self._insert_wait_leg(legs, best.duration_s)

        legs_distance = sum(leg.distance_m for leg in legs if leg.mode != "wait")
        total_distance = (
            legs_distance if legs_distance > best.distance_m * 1.5
            else best.distance_m
        )

        return IntermodalRoute(
            label="🚌 Public Transport", strategy="pt_direct",
            legs=legs,
            total_duration_s=best.duration_s,
            total_distance_m=total_distance,
            transfers=best.transfers,
            geometry=best.geometry or best.points or {},
        )

    def _first_mile_gated(
        self,
        from_lat: float, from_lon: float,
        stop: BoardingStop,
        to_lat: float, to_lon: float,
        first_mode: str, label: str, strategy: str,
        dep_dt: datetime,
        is_transfer_elimination: bool,
    ) -> Optional[IntermodalRoute]:
        """
        First-mile route with the genuine-advantage gate applied.

        The gate checks:
          1. First-mile distance ≥ minimum for the mode
          2. First-mile time saving vs walking ≥ min_saving_s
          3. For transfer elimination: confirms we actually skip a transfer

        Only proceeds to GraphHopper routing if the gate passes.
        """
        p = self.policy

        # ── Gate 1: minimum first-mile distance ──────────────────────────
        dist_to_stop_m = self._haversine_km(
            from_lat, from_lon, stop.lat, stop.lon
        ) * 1000

        min_dist = (
            p.bike_first_mile_min_m if first_mode == "bike"
            else p.car_first_mile_min_m
        )
        if dist_to_stop_m < min_dist:
            return None  # stop is too close — just walk there (pt_direct handles this)

        # ── Gate 2: time saving vs walking ───────────────────────────────
        walk_time_s = dist_to_stop_m / p.walk_speed_ms
        mode_speed = p.bike_speed_ms if first_mode == "bike" else 10.0  # ~36 km/h car urban avg
        estimated_mode_time_s = dist_to_stop_m / mode_speed
        time_saving_s = walk_time_s - estimated_mode_time_s

        if time_saving_s < p.first_mile_min_saving_s:
            return None  # mode isn't meaningfully faster than walking

        # ── Gate passed — proceed with actual routing ────────────────────
        route = self._combine_first_mile(
            from_lat, from_lon,
            stop.lat, stop.lon, stop.name, stop.departure,
            to_lat, to_lon,
            first_mode, label, strategy, dep_dt,
        )

        if route is not None:
            route.first_mile_saving_s = time_saving_s
            if is_transfer_elimination and stop.transfer_index >= 1:
                route.transfers_eliminated = stop.transfer_index
                # Update label to make transfer elimination visible in UI
                route.label = label  # already set to "transfer-free" variant

        return route

    def _combine_first_mile(
        self,
        from_lat: float, from_lon: float,
        stop_lat: float, stop_lon: float,
        stop_name: str, stop_departure: Optional[str],
        to_lat: float, to_lon: float,
        first_mode: str, label: str, strategy: str,
        dep_dt: datetime,
    ) -> Optional[IntermodalRoute]:
        """Route first-mode leg to stop, then PT from stop to destination."""
        first_routes = getattr(self.client, f"route_{first_mode}")(
            from_lat, from_lon, stop_lat, stop_lon
        )
        if not first_routes:
            return None
        first = first_routes[0]

        # PT departure = trip departure + first-mile travel + boarding buffer
        # NOTE: first_mile_penalty_min is applied ONCE here only.
        pt_departure = dep_dt + timedelta(
            seconds=first.duration_s + self.policy.first_mile_penalty_min * 60
        )

        pt_routes = self.client.route_pt(
            stop_lat, stop_lon, to_lat, to_lon,
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
        overhead_s = self.policy.car_overhead_min * 60 if first_mode == "car" else 0

        first_leg = IntermodalLeg(
            mode=first_mode,
            description=f"{icon} {first_mode.title()} to {stop_name}",
            distance_m=first.distance_m,
            duration_s=first.duration_s + overhead_s,
            from_name="Origin",
            to_name=stop_name,
            geometry=first.geometry,
        )

        pt_legs = self._convert_pt_legs(pt)
        pt_legs = self._insert_wait_leg(pt_legs, pt.duration_s)

        legs = [first_leg] + pt_legs
        total_duration = first.duration_s + overhead_s + pt.duration_s
        total_distance = first.distance_m + sum(
            l.distance_m for l in pt_legs if l.mode != "wait"
        )

        return IntermodalRoute(
            label=label, strategy=strategy,
            legs=legs,
            total_duration_s=total_duration,
            total_distance_m=total_distance,
            transfers=pt.transfers,
            geometry=self._merge_leg_geometries(legs) or pt.geometry or pt.points or {},
        )

    def _egress_substitution(
        self,
        from_lat: float, from_lon: float,
        to_lat: float, to_lon: float,
        egress_mode: str, label: str, strategy: str,
        dep_dt: datetime,
        direct_pt_routes: list,
    ) -> Optional[IntermodalRoute]:
        """
        Substitute the final PT egress walk with a bike leg when the
        egress walk is long enough to make cycling worthwhile.
        """
        p = self.policy
        if not direct_pt_routes:
            return None

        best_pt = min(
            [r for r in direct_pt_routes if any(isinstance(l, PtLeg) for l in r.legs)],
            key=lambda r: r.duration_s,
            default=None,
        )
        if best_pt is None:
            return None

        # Find the final alighting stop
        last_pt_leg = next(
            (l for l in reversed(best_pt.legs) if isinstance(l, PtLeg)),
            None,
        )
        if last_pt_leg is None or not last_pt_leg.stops:
            return None

        # Get coordinates of final alighting stop
        final_stop = last_pt_leg.stops[-1]
        lon, lat = self._stop_lon_lat(final_stop)
        if lat is None or lon is None:
            return None

        # Egress walk distance
        egress_m = self._haversine_km(lat, lon, to_lat, to_lon) * 1000
        if egress_m < p.egress_min_m:
            return None  # egress too short — walk is fine
        if egress_m > p.bike_egress_max_m:
            return None  # egress too long for cycling too

        # Route PT from origin to the final alighting stop
        pt_routes = self.client.route_pt(
            from_lat, from_lon, lat, lon,
            departure_time=dep_dt,
            max_walk_meters=self.max_walk,
            limit_solutions=2,
        )
        if not pt_routes:
            return None

        pt_candidates = [r for r in pt_routes if any(isinstance(l, PtLeg) for l in r.legs)]
        if not pt_candidates:
            return None

        pt = min(pt_candidates, key=lambda r: r.duration_s)

        # Bike from alighting stop to destination
        bike_routes = self.client.route_bike(lat, lon, to_lat, to_lon)
        if not bike_routes:
            return None
        bike = bike_routes[0]

        pt_legs = self._convert_pt_legs(pt)
        pt_legs = self._insert_wait_leg(pt_legs, pt.duration_s)

        egress_leg = IntermodalLeg(
            mode=egress_mode,
            description=f"🚴 Bike to destination",
            distance_m=bike.distance_m,
            duration_s=bike.duration_s,
            from_name=final_stop.get("stop_name", "PT Stop"),
            to_name="Destination",
            geometry=bike.geometry,
        )

        legs = pt_legs + [egress_leg]
        total_duration = pt.duration_s + bike.duration_s
        total_distance = pt.distance_m + bike.distance_m

        return IntermodalRoute(
            label=label, strategy=strategy,
            legs=legs,
            total_duration_s=total_duration,
            total_distance_m=total_distance,
            transfers=pt.transfers,
            geometry=self._merge_leg_geometries(legs) or {},
        )

    # ──────────────────────────────────────────────────────────────────────
    # Boarding stop discovery (first boarding + transfer elimination stops)
    # ──────────────────────────────────────────────────────────────────────

    def _find_all_boarding_stops(
        self,
        from_lat: float, from_lon: float,
        to_lat: float, to_lon: float,
        dep_dt: datetime,
        direct_pt_routes: list,
    ) -> list[BoardingStop]:
        """
        Return all candidate boarding stops:
          - transfer_index=0: first boarding stops (standard first-mile)
          - transfer_index≥1: post-transfer boarding stops (transfer elimination)

        For a PT route Walk→TramA→Transfer→TramB→Walk, this returns:
          - The first stop of TramA leg  (transfer_index=0)
          - The first stop of TramB leg  (transfer_index=1)  ← transfer elimination
        """
        cache_key = (
            round(from_lat, 3), round(from_lon, 3),
            round(to_lat, 3), round(to_lon, 3),
            dep_dt.date().isoformat(),
        )
        if cache_key in self._stop_cache:
            return self._stop_cache[cache_key]

        stops: list[BoardingStop] = []
        seen: set = set()

        for pt_route in direct_pt_routes:
            pt_legs = [l for l in pt_route.legs if isinstance(l, PtLeg)]
            for pt_leg_idx, pt_leg in enumerate(pt_legs):
                if not pt_leg.stops:
                    continue
                first_stop = pt_leg.stops[0]
                lon, lat = self._stop_lon_lat(first_stop)
                name = first_stop.get("stop_name", "PT Stop")

                if lat is None or lon is None:
                    continue

                key = (round(lat, 5), round(lon, 5))
                if key in seen:
                    continue
                seen.add(key)

                # Estimate walk time from origin to this stop
                walk_m = self._haversine_km(from_lat, from_lon, lat, lon) * 1000
                walk_time_s = walk_m / self.policy.walk_speed_ms

                stops.append(BoardingStop(
                    lat=float(lat),
                    lon=float(lon),
                    name=name,
                    departure=first_stop.get("departure_time"),
                    transfer_index=pt_leg_idx,   # 0=first PT leg, 1=after first transfer, etc.
                    walk_time_s=walk_time_s,
                    pt_leg_index=pt_leg_idx,
                ))

        # Sort: first-boarding stops first, then transfer-elimination stops
        stops.sort(key=lambda s: (s.transfer_index, s.walk_time_s))

        self._stop_cache[cache_key] = stops
        return stops

    def _cached_pt_routes(
        self,
        from_lat: float, from_lon: float,
        to_lat: float, to_lon: float,
        dep_dt: datetime,
    ) -> list:
        """Fetch direct PT routes with caching."""
        cache_key = (
            round(from_lat, 4), round(from_lon, 4),
            round(to_lat, 4), round(to_lon, 4),
            dep_dt.isoformat()[:16],
        )
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        try:
            routes = self.client.route_pt(
                from_lat, from_lon, to_lat, to_lon,
                departure_time=dep_dt,
                max_walk_meters=self.policy.stop_search_walk_m,
                limit_solutions=3,
            )
        except Exception:
            routes = []

        self._route_cache[cache_key] = routes
        return routes

    @staticmethod
    def _best_pt_duration(pt_routes: list) -> Optional[float]:
        """Return the duration of the best direct PT route."""
        valid = [r for r in pt_routes if any(isinstance(l, PtLeg) for l in r.legs)]
        if not valid:
            return None
        return min(r.duration_s for r in valid)

    # ──────────────────────────────────────────────────────────────────────
    # Route realism evaluation
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate_route_realism(
        self,
        route: IntermodalRoute,
        distance_km: float,
        agent: Optional[Agent] = None,
    ) -> None:
        """Update route.feasible, generalized_cost_s, realism_score, decision_reason."""
        if not route.feasible:
            if not route.decision_reason:
                route.decision_reason = route.infeasible_reason
            self._compute_generalized_cost(route)
            return

        max_transfers = self._max_transfers_for_distance(distance_km, agent)

        if route.transfers > max_transfers:
            route.feasible = False
            route.infeasible_reason = (
                f"Too many transfers for {distance_km:.1f} km trip: "
                f"{route.transfers} > allowed {max_transfers}."
            )
            route.decision_reason = route.infeasible_reason
            self._compute_generalized_cost(route)
            return

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
        route.decision_reason = self._build_decision_reason(
            route, distance_km, max_transfers
        )

    def _compute_generalized_cost(self, route: IntermodalRoute) -> None:
        """
        Generalised journey cost for sorting.

        NOTE: first_mile_penalty is NOT added here — it is already baked
        into total_duration_s via _combine_first_mile. Adding it again
        would double-count it.
        """
        transfer_penalty_s = max(0, route.transfers) * self.policy.transfer_penalty_min * 60
        boarding_penalty_s = 0
        if any(leg.mode == "pt" for leg in route.legs):
            boarding_penalty_s += self.policy.pt_boarding_penalty_min * 60

        route.generalized_cost_s = (
            route.total_duration_s + transfer_penalty_s + boarding_penalty_s
        )

    def _compute_realism_score(
        self, route: IntermodalRoute, distance_km: float
    ) -> float:
        score = 1.0
        score -= min(0.45, 0.15 * max(0, route.transfers))
        if distance_km <= self.policy.very_short_km and route.strategy in (
            "pt_direct", "bike_pt", "car_pt"
        ):
            score -= 0.4
        walking_m = sum(l.distance_m for l in route.legs if l.mode == "walk")
        if walking_m > self.max_walk * 2:
            score -= 0.2
        if route.strategy in ("bike_pt", "car_pt", "bike_pt_elim", "car_pt_elim"):
            if route.transfers >= 2:
                score -= 0.15
        if route.transfers_eliminated > 0:
            score += 0.1 * route.transfers_eliminated  # reward transfer elimination
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
            f"gen_cost={int(route.generalized_cost_s // 60)}min",
        ]
        if route.transfers_eliminated > 0:
            parts.append(f"eliminated {route.transfers_eliminated} transfer(s)")
        if route.first_mile_saving_s > 0:
            parts.append(f"first_mile_saving={int(route.first_mile_saving_s//60)}min")
        if route.strategy in ("bike_pt", "car_pt", "bike_pt_elim", "car_pt_elim"):
            parts.append("intermodal accepted")
        elif route.strategy == "bike_egress":
            parts.append("egress substitution accepted")
        elif route.strategy == "pt_direct":
            parts.append("direct PT accepted")
        else:
            parts.append("direct mode accepted")
        return " | ".join(parts)

    # ──────────────────────────────────────────────────────────────────────
    # Feasibility checks
    # ──────────────────────────────────────────────────────────────────────

    def check_feasibility(
        self, route: IntermodalRoute, agent: Agent
    ) -> tuple[bool, str]:
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

        if route.strategy in ("bike_pt", "bike_pt_elim"):
            bike_km = (
                sum(l.distance_m for l in route.legs if l.mode == "bike") / 1000
            )
            if bike_km > limits["bike"]:
                return False, (
                    f"Bike leg {bike_km:.1f}km exceeds practical limit "
                    f"({limits['bike']}km for {profile_type} profile)."
                )

        return True, ""

    @staticmethod
    def _profile_limits(profile_type: str) -> dict[str, float]:
        defaults = {"walk": 4.5, "bike": 6.5}
        table = {
            "biospheric": {"walk": 6.0, "bike": 8.0},
            "altruistic":  {"walk": 4.5, "bike": 6.5},
            "egoistic":    {"walk": 3.0, "bike": 6.0},
            "hedonic":     {"walk": 2.5, "bike": 5.0},
        }
        return table.get(profile_type, defaults)

    # ──────────────────────────────────────────────────────────────────────
    # PT leg conversion and geometry helpers
    # ──────────────────────────────────────────────────────────────────────

    def _convert_pt_legs(self, route: Route) -> list[IntermodalLeg]:
        result: list[IntermodalLeg] = []
        for gh_leg in route.legs:
            if isinstance(gh_leg, WalkLeg):
                result.append(IntermodalLeg(
                    mode="walk", description="🚶 Walk",
                    distance_m=gh_leg.distance_m,
                    duration_s=gh_leg.duration_s,
                    geometry=gh_leg.geometry,
                ))
            elif isinstance(gh_leg, PtLeg):
                distance_m = self._distance_from_geometry_or_stops(
                    gh_leg.geometry, gh_leg.stops
                )
                result.append(IntermodalLeg(
                    mode="pt",
                    description=f"🚌 {gh_leg.route_id}" if gh_leg.route_id else "🚌 PT",
                    distance_m=distance_m,
                    duration_s=gh_leg.duration_s,
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

    def _insert_wait_leg(
        self, legs: list[IntermodalLeg], total_duration_s: float
    ) -> list[IntermodalLeg]:
        """Insert a wait leg after the first access-walk leg if there is measurable wait."""
        leg_sum_s = sum(getattr(l, "duration_s", 0) for l in legs)
        wait_s = max(0.0, total_duration_s - leg_sum_s)
        if wait_s <= 30:
            return legs
        first_walks = [i for i, l in enumerate(legs) if l.mode == "walk"]
        insert_after = first_walks[0] + 1 if first_walks else 0
        wait_leg = IntermodalLeg(
            mode="wait", description="⏳ Waiting at stop",
            distance_m=0, duration_s=wait_s,
        )
        return legs[:insert_after] + [wait_leg] + legs[insert_after:]

    def _distance_from_geometry_or_stops(
        self, geometry: Optional[dict], stops: list
    ) -> float:
        coords = []
        if geometry and isinstance(geometry, dict) and geometry.get("coordinates"):
            coords = geometry["coordinates"]
        elif stops:
            for stop in stops:
                sg = stop.get("geometry", {})
                if sg and sg.get("coordinates"):
                    coords.append(sg["coordinates"])
        if len(coords) < 2:
            return 0.0
        distance_m = 0.0
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i][:2]
            lon2, lat2 = coords[i + 1][:2]
            distance_m += self._haversine_km(lat1, lon1, lat2, lon2) * 1000
        return distance_m

    def _merge_leg_geometries(
        self, legs: list[IntermodalLeg]
    ) -> Optional[dict]:
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

    def _deduplicate_strategies(
        self, routes: list[IntermodalRoute]
    ) -> list[IntermodalRoute]:
        """
        When both transfer-elimination and standard variants of bike_pt/car_pt
        are feasible, keep only the better one per base strategy.
        """
        strategy_groups: dict[str, list[IntermodalRoute]] = {}
        for route in routes:
            base = route.strategy.replace("_elim", "")
            strategy_groups.setdefault(base, []).append(route)

        result = []
        for base, group in strategy_groups.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                # Keep the one with the lowest generalized cost
                best = min(group, key=lambda r: (
                    not r.feasible,
                    r.generalized_cost_s if r.generalized_cost_s > 0
                    else r.total_duration_s,
                ))
                result.append(best)

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Distance and utility helpers
    # ──────────────────────────────────────────────────────────────────────

    def _estimate_routable_distance_km(
        self, from_lat: float, from_lon: float,
        to_lat: float, to_lon: float,
    ) -> float:
        crow_km = self._haversine_km(from_lat, from_lon, to_lat, to_lon)
        try:
            walk_routes = self.client.route_foot(from_lat, from_lon, to_lat, to_lon)
            if walk_routes:
                return walk_routes[0].distance_m / 1000
        except Exception:
            pass
        return crow_km

    @staticmethod
    def _haversine_km(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _stop_lon_lat(
        stop: dict,
    ) -> tuple[Optional[float], Optional[float]]:
        geom = stop.get("geometry", {})
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        if coords and len(coords) >= 2:
            return coords[0], coords[1]
        lon = stop.get("lon", stop.get("stop_lon"))
        lat = stop.get("lat", stop.get("stop_lat"))
        return lon, lat

    @staticmethod
    def _need(agent: Optional[Agent], dim: str) -> float:
        """Return agent's need weight for dim, defaulting to 0.5 if no agent."""
        if agent is None:
            return 0.5
        return float(getattr(agent, "value_weights", {}).get(dim, 0.5))

    @staticmethod
    def _agent_can(agent: Optional[Agent], mode: str) -> bool:
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

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set = set()
        out = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out
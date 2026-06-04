"""
personalised_router.py
──────────────────────
The core scoring engine.

For each candidate route it computes a VALUE SCORE by:

  1. Start with the mode's base attribute scores (from value_model.py)
  2. Blend in route-metric adjustments computed from real GraphHopper data
     (actual duration, distance per leg, transfers, mode mix)
  3. Dot-product with the agent's normalised need weights
  4. Normalise to [0, 100] across all candidates for readability

All 11 need dimensions from the cognitive passport are covered:
  pro_env, physical, privacy, autonomy, cost, speed,
  safety_accident, safety_crime, comfort, reliable, health_infection

METRIC_BLEND = 0.7 means real route data drives 70% of the score;
the static mode matrix provides a 30% structural prior.
"""

from dataclasses import dataclass, field
from typing import Optional

from agent import Agent
from value_model import (
    VALUE_DIMENSIONS, MODE_ATTRIBUTES, MODE_LABELS,
    speed_score_from_duration, cost_score_from_mode,
    comfort_score_from_transfers,
    walking_distance_penalty, cycling_distance_penalty,
)
from intermodal_router import IntermodalRouter, IntermodalRoute
from graphhopper_client import GraphHopperClient


# ----------------------------------------------
#  Result dataclasses
# ----------------------------------------------

@dataclass
class DimensionScore:
    """Score for one need dimension for one route."""
    dimension: str
    agent_weight: float       # how much this agent cares (0–1)
    mode_attribute: float     # static mode prior (-1 to +1)
    metric_adjustment: float  # real-route adjustment (-1 to +1)
    blended_attribute: float  # blended final attribute (-1 to +1)
    contribution: float       # agent_weight × blended_attribute


@dataclass
class ScoredRoute:
    """A route with full value-based scoring."""
    route: IntermodalRoute
    mode_key: str
    mode_label: str
    utility_score: float                    # 0–100 final normalised score
    raw_score: float                        # pre-normalisation dot product
    dimension_scores: list[DimensionScore]  # per-dimension breakdown
    rank: int = 0
    available: bool = True
    poi_boost: float = 0.0
    matched_pois: list = field(default_factory=list)

    @property
    def top_matching_values(self) -> list[DimensionScore]:
        positive = [d for d in self.dimension_scores if d.contribution > 0]
        return sorted(positive, key=lambda d: d.contribution, reverse=True)[:3]

    @property
    def top_conflicting_values(self) -> list[DimensionScore]:
        negative = [d for d in self.dimension_scores if d.contribution < 0]
        return sorted(negative, key=lambda d: d.contribution)[:2]


# ----------------------------------------------
#  Personalised Router
# ----------------------------------------------

class PersonalisedRouter:

    # Route metrics drive 70% of the score; static matrix = 30% prior
    METRIC_BLEND = 0.7

    def __init__(self, client: GraphHopperClient,
                 pois: Optional[list] = None,
                 poi_proximity_m: float = 100):
        self.client          = client
        self.pois            = pois or []
        self.poi_proximity_m = poi_proximity_m

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------

    def route(self, agent: Agent,
              from_lat: float, from_lon: float,
              to_lat: float,   to_lon: float,
              departure: Optional[str] = None,
              max_walk_m: int = 500) -> list[ScoredRoute]:

        im_router = IntermodalRouter(
            client     = self.client,
            departure  = departure,
            max_walk_m = max_walk_m,
        )
        candidate_routes = im_router.plan(from_lat, from_lon, to_lat, to_lon)

        scored = []
        for route in candidate_routes:
            if not route.feasible:
                continue

            mode_key = self._strategy_to_mode_key(route.strategy)
            if mode_key not in MODE_ATTRIBUTES:
                continue

            is_feasible, reason = im_router.check_feasibility(route, agent)
            if not is_feasible:
                route.feasible          = False
                route.infeasible_reason = reason
                continue

            scored_route          = self._score_route(agent, route, mode_key)
            available_modes       = agent.available_modes()
            scored_route.available = (
                agent.can_use(mode_key.split("_")[0])
                or mode_key in available_modes
                or mode_key == "foot"
            )
            scored.append(scored_route)

        if not scored:
            return []

        # Normalise raw scores to 0–100
        raw_scores = [s.raw_score for s in scored]
        min_raw    = min(raw_scores)
        max_raw    = max(raw_scores)
        span       = max_raw - min_raw if max_raw != min_raw else 1.0

        for sr in scored:
            sr.utility_score = round(((sr.raw_score - min_raw) / span) * 100, 1)

        scored.sort(key=lambda s: s.utility_score, reverse=True)
        for i, sr in enumerate(scored, 1):
            sr.rank = i

        return scored

    # ------------------------------------------------------------------
    #  Scoring
    # ------------------------------------------------------------------

    def _score_route(self, agent: Agent,
                     route: IntermodalRoute,
                     mode_key: str) -> ScoredRoute:

        base_attrs     = MODE_ATTRIBUTES[mode_key]
        metric_adjusts = self._metric_adjustments(route, mode_key, agent)
        dim_scores     = []
        raw_total      = 0.0

        for dim in VALUE_DIMENSIONS:
            agent_weight = agent.value_weights.get(dim, 0.0)
            mode_attr    = base_attrs.get(dim, 0.0)
            metric_adj   = metric_adjusts.get(dim, 0.0)

            blended = (mode_attr    * (1 - self.METRIC_BLEND)
                       + metric_adj *      self.METRIC_BLEND)
            blended = max(-1.0, min(1.0, blended))

            contribution = agent_weight * blended
            raw_total   += contribution

            dim_scores.append(DimensionScore(
                dimension         = dim,
                agent_weight      = agent_weight,
                mode_attribute    = mode_attr,
                metric_adjustment = metric_adj,
                blended_attribute = blended,
                contribution      = contribution,
            ))

        poi_boost, matched_pois = 0.0, []
        if self.pois:
            poi_boost, matched_pois = self.compute_poi_score(
                route, self.pois, agent, self.poi_proximity_m
            )
            raw_total += poi_boost

        return ScoredRoute(
            route            = route,
            mode_key         = mode_key,
            mode_label       = MODE_LABELS.get(mode_key, mode_key),
            utility_score    = 0.0,
            raw_score        = raw_total,
            dimension_scores = dim_scores,
            poi_boost        = poi_boost,
            matched_pois     = matched_pois,
        )

    def _metric_adjustments(self, route: IntermodalRoute,
                             mode_key: str, agent: Agent) -> dict:
        """
        Compute all 11 need-dimension scores from real route data.
        Returns {dimension: float in [-1, +1]}.
        """
        adj          = {}
        distance_km  = route.total_distance_m / 1000
        duration_s   = route.total_duration_s
        profile_type = agent.infer_profile_type()
        legs         = route.legs or []

        # ── Leg distance breakdown ────────────────────────────────────
        walk_m  = sum(l.distance_m for l in legs if l.mode == "walk")
        bike_m  = sum(l.distance_m for l in legs if l.mode == "bike")
        car_m   = sum(l.distance_m for l in legs if l.mode == "car")
        pt_m    = sum(l.distance_m for l in legs if l.mode == "pt")
        active_m    = walk_m + bike_m
        total_m     = max(route.total_distance_m, 1.0)

        frac_active = active_m / total_m
        frac_car    = car_m    / total_m
        frac_pt     = pt_m     / total_m

        # ── pro_env ───────────────────────────────────────────────────
        # CO₂ proxy: car ≈ 120 g/km, PT ≈ 40 g/km, active ≈ 0
        co2_g          = (car_m / 1000 * 120) + (pt_m / 1000 * 40)
        co2_normalised = min(1.0, co2_g / 1200)   # 10 km car = max dirty
        adj["pro_env"] = 1.0 - 2.0 * co2_normalised

        # ── physical ──────────────────────────────────────────────────
        # Active distance: 5 km = +1, 0 km = 0; capped + exertion penalty
        active_km      = active_m / 1000
        raw_physical   = min(1.0, active_km / 5.0)
        penalty        = (walking_distance_penalty(walk_m / 1000, profile_type)
                          + cycling_distance_penalty(bike_m / 1000, profile_type))
        adj["physical"] = max(-1.0, raw_physical + penalty * 0.3)

        # ── privacy ───────────────────────────────────────────────────
        adj["privacy"] = max(-1.0, min(1.0,
            frac_car  *  1.0 +
            frac_pt   * -0.8 +
            frac_active * 0.6
        ))

        # ── autonomy ──────────────────────────────────────────────────
        # Car + active = agent controls pace; PT = schedule-bound
        frac_autonomous = frac_car + frac_active
        adj["autonomy"] = max(-1.0, min(1.0, frac_autonomous * 2.0 - 1.0))

        # ── cost ──────────────────────────────────────────────────────
        adj["cost"] = cost_score_from_mode(mode_key, route.total_distance_m)

        # ── speed ─────────────────────────────────────────────────────
        base_speed = speed_score_from_duration(duration_s, reference_s=1800)
        if mode_key == "foot":
            base_speed += walking_distance_penalty(distance_km, profile_type) * 0.5
        elif mode_key == "bike":
            base_speed += cycling_distance_penalty(distance_km, profile_type) * 0.4
        adj["speed"] = max(-1.0, min(1.0, base_speed))

        # ── safety_accident ───────────────────────────────────────────
        # Per-mode accident risk weights; bike most exposed on road
        ACCIDENT_RISK = {"walk": 0.15, "bike": 0.55, "car": 0.30, "pt": 0.10}
        if legs:
            weighted_risk = sum(
                (l.distance_m / total_m) * ACCIDENT_RISK.get(l.mode, 0.2)
                for l in legs
            )
        else:
            weighted_risk = ACCIDENT_RISK.get(mode_key.split("_")[0], 0.2)
        adj["safety_accident"] = max(-1.0, min(1.0, 1.0 - 2.0 * weighted_risk))

        # ── safety_crime ──────────────────────────────────────────────
        # Car = safest (enclosed), walking outdoors = most exposed
        adj["safety_crime"] = max(-1.0, min(1.0,
            frac_car    *  0.8 +
            frac_pt     *  0.0 +
            frac_active * -0.3
        ))

        # ── comfort ───────────────────────────────────────────────────
        transfer_sc   = comfort_score_from_transfers(route.transfers)
        exertion_pen  = (walking_distance_penalty(walk_m / 1000, profile_type) * 0.5
                         + cycling_distance_penalty(bike_m / 1000, profile_type) * 0.3)
        mode_comfort  = frac_car * 0.8 + frac_pt * 0.2 + frac_active * -0.2
        adj["comfort"] = max(-1.0, min(1.0,
            transfer_sc  * 0.4
            + exertion_pen * 0.3
            + mode_comfort * 0.3
        ))

        # ── reliable ──────────────────────────────────────────────────
        schedule_penalty  = frac_pt * -0.6
        transfer_penalty  = min(0.0, -0.15 * route.transfers)
        adj["reliable"] = max(-1.0, min(1.0,
            (1.0 - frac_pt) * 0.6 + schedule_penalty + transfer_penalty
        ))

        # ── health_infection ──────────────────────────────────────────
        # PT = enclosed crowded = high risk; outdoor modes = low risk
        adj["health_infection"] = max(-1.0, min(1.0, 1.0 - 2.0 * frac_pt))

        return adj

    # ------------------------------------------------------------------
    #  POI scoring (foundation for future extension)
    # ------------------------------------------------------------------

    def compute_poi_score(self, route: IntermodalRoute, pois: list,
                          agent: Agent,
                          proximity_m: float = 100) -> tuple[float, list]:
        total_boost  = 0.0
        matched_pois = []
        route_coords = self._extract_route_coordinates(route)
        if not route_coords:
            return 0.0, []

        for poi in pois:
            poi_lat   = poi.get("lat")
            poi_lon   = poi.get("lon")
            alignment = poi.get("value_alignment", {})
            if not poi_lat or not poi_lon or not alignment:
                continue

            min_dist = min(
                self._haversine_distance(poi_lat, poi_lon, lat, lon)
                for lat, lon in route_coords
            )
            if min_dist <= proximity_m:
                boost = sum(
                    agent.value_weights.get(dim, 0.0) * alignment.get(dim, 0.0)
                    for dim in alignment
                )
                total_boost += boost
                matched_pois.append({
                    "name":       poi.get("name", "Unknown POI"),
                    "type":       poi.get("type", "unknown"),
                    "boost":      boost,
                    "distance_m": min_dist,
                })

        return total_boost, matched_pois

    def _extract_route_coordinates(self, route: IntermodalRoute):
        coords   = []
        geometry = route.geometry
        if not geometry:
            for leg in route.legs:
                if hasattr(leg, "stops") and leg.stops:
                    for stop in leg.stops:
                        sg = stop.get("geometry", {})
                        if sg and "coordinates" in sg:
                            lon, lat = sg["coordinates"]
                            coords.append((lat, lon))
            return coords
        if isinstance(geometry, dict) and "coordinates" in geometry:
            for coord in geometry["coordinates"]:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    coords.append((coord[1], coord[0]))
        return coords

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2) -> float:
        import math
        R    = 6371000
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a    = (math.sin(dlat / 2) ** 2
                + math.cos(math.radians(lat1))
                * math.cos(math.radians(lat2))
                * math.sin(dlon / 2) ** 2)
        return R * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def _strategy_to_mode_key(strategy: str) -> str:
        return {
            "pt_direct":   "pt",
            "car_direct":  "car",
            "bike_direct": "bike",
            "foot_direct": "foot",
            "bike_pt":     "bike_pt",
            "car_pt":      "car_pt",
        }.get(strategy, strategy)
"""
value_model.py
──────────────
Defines the VALUE ATTRIBUTE MATRIX — how well each transport mode/strategy
satisfies each human value dimension.

Scores are in the range [-1.0, +1.0]:
  +1.0  = this mode strongly satisfies this value
   0.0  = neutral
  -1.0  = this mode strongly conflicts with this value

This file also defines distance-based feasibility curves for transport modes.
These curves model behavioral plausibility as a smooth function of trip distance,
instead of using hard thresholds.

The nine value dimensions come from the psychological model output:
  pro_environment   — environmental concern
  physical_activity — desire for physical exercise
  privacy           — preference for personal space / no crowding
  autonomy          — preference for self-directed travel
  hedonism          — enjoyment / pleasure of the journey itself
  cost_saving       — sensitivity to monetary cost
  speed             — preference for fastest journey
  safety            — concern about personal safety
  comfort           — preference for comfortable, stress-free travel
"""

from dataclasses import dataclass
import math


# ----------------------------------------------
#  Value dimensions
# ----------------------------------------------

VALUE_DIMENSIONS = [
    "pro_environment",
    "physical_activity",
    "privacy",
    "autonomy",
    "hedonism",
    "cost_saving",
    "speed",
    "safety",
    "comfort",
]


# ----------------------------------------------
#  Mode attribute matrix
#  Each entry is a dict of {value_dimension: score}.
#  Missing dimensions default to 0.0.
# ----------------------------------------------

MODE_ATTRIBUTES = {

    "foot": {
        "pro_environment":   1.0,
        "physical_activity": 1.0,
        "privacy":           0.8,
        "autonomy":          0.8,
        "hedonism":          0.4,
        "cost_saving":       1.0,
        "speed":            -1.0,
        "safety":            0.3,
        "comfort":          -0.5,
    },

    "bike": {
        "pro_environment":   1.0,
        "physical_activity": 0.9,
        "privacy":           0.8,
        "autonomy":          0.9,
        "hedonism":          0.6,
        "cost_saving":       0.8,
        "speed":             0.2,
        "safety":           -0.2,
        "comfort":          -0.2,
    },

    "car": {
        "pro_environment":  -1.0,
        "physical_activity":-1.0,
        "privacy":          1.0,
        "autonomy":         1.0,
        "hedonism":         0.3,
        "cost_saving":     -0.8,
        "speed":            0.9,
        "safety":           0.2,
        "comfort":          0.9,
    },

    "pt": {
        "pro_environment":   0.8,
        "physical_activity": 0.1,
        "privacy":          -0.8,
        "autonomy":         -0.8,
        "hedonism":         -0.1,
        "cost_saving":       0.6,
        "speed":             0.3,
        "safety":            0.8,
        "comfort":           0.2,
    },

    "bike_pt": {
        "pro_environment":   0.9,
        "physical_activity": 0.6,
        "privacy":          -0.2,
        "autonomy":          0.1,
        "hedonism":          0.3,
        "cost_saving":       0.7,
        "speed":             0.5,
        "safety":            0.2,
        "comfort":          -0.1,
    },

    "car_pt": {
        "pro_environment":  -0.2,
        "physical_activity":-0.5,
        "privacy":          0.2,
        "autonomy":         0.3,
        "hedonism":         0.2,
        "cost_saving":     -0.1,
        "speed":            0.8,
        "safety":           0.5,
        "comfort":          0.7,
    },
}


# ----------------------------------------------
#  Human-readable labels
# ----------------------------------------------

MODE_LABELS = {
    "foot":    "🚶 Walk",
    "bike":    "🚴 Bike",
    "car":     "🚗 Car",
    "pt":      "🚌 Public Transport",
    "bike_pt": "🚴+🚌 Bike & PT",
    "car_pt":  "🚗+🚌 Car & PT (P&R)",
}


# ----------------------------------------------
#  Required beliefs per mode
# ----------------------------------------------

MODE_BELIEF_REQUIREMENTS = {
    "foot":    [],
    "bike":    ["owns_bike"],
    "car":     ["owns_car"],
    "pt":      ["has_pt_access"],
    "bike_pt": ["owns_bike", "has_pt_access"],
    "car_pt":  ["owns_car", "has_pt_access"],
}


# =============================================================================
# Distance-Based Feasibility Curves
# =============================================================================

@dataclass(frozen=True)
class DistanceFeasibilityParams:
    """
    Parameters for distance-based behavioral feasibility curves.

    Distance is measured in kilometres.

    The defaults correspond to the baseline model:
      - Walk: logistic decay
      - Bike: Gaussian peak
      - Car: logistic rise
      - PT: hump-shaped rise-and-decay curve
    """

    # Walking logistic decay
    walk_d0_km: float = 2.35
    walk_k: float = 2.57

    # Cycling Gaussian peak
    bike_mu_km: float = 3.0
    bike_sigma_km: float = 1.5

    # Car logistic rise
    car_d0_km: float = 4.5
    car_k: float = 0.92

    # Public transport hump-shaped curve
    pt_rise_k: float = 1.0
    pt_rise_d0_km: float = 2.5
    pt_decay_k: float = 0.25
    pt_decay_d0_km: float = 12.0

    # Transfer feasibility
    transfer_alpha: float = 0.55
    transfer_distance_tolerance_base: float = 0.75
    transfer_distance_tolerance_slope: float = 0.18

    # Numerical stability
    epsilon: float = 1e-6


DEFAULT_FEASIBILITY_PARAMS = DistanceFeasibilityParams()


def _clamp01(x: float) -> float:
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, float(x)))


def walk_distance_feasibility(
    distance_km: float,
    params: DistanceFeasibilityParams = DEFAULT_FEASIBILITY_PARAMS,
) -> float:
    """
    Walking feasibility: logistic decay.

    High for short trips, then rapidly declines after the midpoint distance.
    """
    d = max(0.0, distance_km)
    value = 1.0 / (1.0 + math.exp(params.walk_k * (d - params.walk_d0_km)))
    return _clamp01(value)


def bike_distance_feasibility(
    distance_km: float,
    params: DistanceFeasibilityParams = DEFAULT_FEASIBILITY_PARAMS,
) -> float:
    """
    Cycling feasibility: Gaussian peak.

    Cycling is most feasible around the medium-distance sweet spot.
    """
    d = max(0.0, distance_km)
    sigma = max(params.bike_sigma_km, params.epsilon)
    value = math.exp(-((d - params.bike_mu_km) ** 2) / (2.0 * sigma ** 2))
    return _clamp01(value)


def car_distance_feasibility(
    distance_km: float,
    params: DistanceFeasibilityParams = DEFAULT_FEASIBILITY_PARAMS,
) -> float:
    """
    Car feasibility: logistic rise.

    Car becomes increasingly feasible as distance increases.
    """
    d = max(0.0, distance_km)
    value = 1.0 / (1.0 + math.exp(-params.car_k * (d - params.car_d0_km)))
    return _clamp01(value)


def pt_distance_feasibility(
    distance_km: float,
    params: DistanceFeasibilityParams = DEFAULT_FEASIBILITY_PARAMS,
) -> float:
    """
    Public transport feasibility: hump-shaped curve.

    PT is weak for very short trips, strongest for medium/long urban trips,
    and then slightly declines for very long trips.
    """
    d = max(0.0, distance_km)
    rise = 1.0 / (1.0 + math.exp(-params.pt_rise_k * (d - params.pt_rise_d0_km)))
    decay = 1.0 / (1.0 + math.exp(params.pt_decay_k * (d - params.pt_decay_d0_km)))
    return _clamp01(rise * decay)


def transfer_feasibility(
    transfers: int,
    distance_km: float,
    params: DistanceFeasibilityParams = DEFAULT_FEASIBILITY_PARAMS,
) -> float:
    """
    Soft transfer feasibility.

    Transfers are more acceptable for longer trips, but they still reduce
    behavioral plausibility.
    """
    n = max(0, int(transfers or 0))
    d = max(0.1, distance_km)
    tolerance = (
        params.transfer_distance_tolerance_base
        + params.transfer_distance_tolerance_slope * d
    )
    value = math.exp(-params.transfer_alpha * n / max(tolerance, params.epsilon))
    return _clamp01(value)


def mode_distance_feasibility(
    mode_key: str,
    distance_km: float,
    transfers: int = 0,
    params: DistanceFeasibilityParams = DEFAULT_FEASIBILITY_PARAMS,
) -> float:
    """
    Distance feasibility for a mode or intermodal strategy.

    Parameters
    ----------
    mode_key:
        One of: foot, bike, car, pt, bike_pt, car_pt.
    distance_km:
        Trip distance in kilometres.
    transfers:
        Number of transfers for PT or intermodal options.

    Returns
    -------
    float
        Behavioral feasibility in [0, 1].
    """
    mode_key = mode_key.lower()

    walk_f = walk_distance_feasibility(distance_km, params)
    bike_f = bike_distance_feasibility(distance_km, params)
    car_f = car_distance_feasibility(distance_km, params)
    pt_f = pt_distance_feasibility(distance_km, params)
    tf = transfer_feasibility(transfers, distance_km, params)

    if mode_key == "foot":
        return walk_f

    if mode_key == "bike":
        return bike_f

    if mode_key == "car":
        return car_f

    if mode_key == "pt":
        return _clamp01(pt_f * tf)

    if mode_key == "bike_pt":
        # Bike+PT combines active first-mile feasibility with PT usefulness.
        return _clamp01((0.45 * bike_f + 0.55 * pt_f) * tf)

    if mode_key == "car_pt":
        # Car+PT combines car access with PT usefulness.
        # A soft park-and-ride gate prevents car+PT from becoming too strong
        # for very short trips.
        park_ride_gate = 1.0 / (1.0 + math.exp(-1.2 * (distance_km - 3.0)))
        return _clamp01((0.35 * car_f + 0.65 * pt_f) * tf * park_ride_gate)

    return 0.0


def all_mode_distance_feasibilities(
    distance_km: float,
    transfers_by_mode: dict | None = None,
    params: DistanceFeasibilityParams = DEFAULT_FEASIBILITY_PARAMS,
) -> dict[str, float]:
    """
    Convenience helper returning feasibility for all supported modes.
    """
    transfers_by_mode = transfers_by_mode or {}
    return {
        mode: mode_distance_feasibility(
            mode,
            distance_km,
            transfers=transfers_by_mode.get(mode, 0),
            params=params,
        )
        for mode in MODE_ATTRIBUTES.keys()
    }


def feasibility_log_term(
    feasibility: float,
    params: DistanceFeasibilityParams = DEFAULT_FEASIBILITY_PARAMS,
) -> float:
    """
    Log feasibility term for utility models.

    This supports utility formulations like:
        U_m = ... + lambda_m * log(F_m(d) + epsilon)

    The output is <= 0, becoming strongly negative when feasibility is near 0.
    """
    return math.log(max(params.epsilon, _clamp01(feasibility)))


# =============================================================================
# Route Metric Scoring Functions
# =============================================================================

def speed_score_from_duration(duration_s: float, reference_s: float = 1800) -> float:
    """
    Convert actual travel time into a speed score.

    reference_s = 30 min baseline. Faster than reference -> positive,
    slower than reference -> negative. Clamped to [-1, +1].
    """
    if reference_s <= 0:
        return 0.0
    ratio = duration_s / reference_s
    score = 1.0 - ratio
    return max(-1.0, min(1.0, score))


def cost_score_from_mode(mode: str, distance_m: float) -> float:
    """
    Estimate relative cost score from mode and distance.

    Returns a score in [-1, +1] where +1 = cheap/free and -1 = expensive.
    """
    distance_km = distance_m / 1000

    if mode == "foot":
        return 1.0
    if mode == "bike":
        return 0.9
    if mode == "pt":
        return 0.5
    if mode == "bike_pt":
        return 0.6
    if mode == "car":
        cost_per_km = 0.30
        relative = min(1.0, distance_km * cost_per_km / 10)
        return -relative
    if mode == "car_pt":
        cost_per_km = 0.15
        relative = min(1.0, distance_km * cost_per_km / 10)
        return -relative * 0.5
    return 0.0


def comfort_score_from_transfers(transfers: int) -> float:
    """More transfers = less comfortable."""
    if transfers <= 0:
        return 0.5
    if transfers == 1:
        return 0.0
    return max(-1.0, -0.3 * transfers)


def walking_distance_penalty(distance_km: float, profile_type: str = "biospheric") -> float:
    """
    Penalty for long walking distances.

    This is retained for compatibility with the current personalised router.
    The new feasibility curves provide a smoother behavioral formulation, while
    this penalty can still act as a route-metric adjustment.
    """
    if distance_km <= 1.0:
        penalty = 0.0
    elif distance_km <= 2.0:
        penalty = -0.3 * (distance_km - 1.0)
    elif distance_km <= 3.0:
        penalty = -0.3 - 0.6 * (distance_km - 2.0)
    elif distance_km <= 5.0:
        penalty = -0.9 - 0.6 * (distance_km - 3.0)
    else:
        penalty = -2.1 - 2.0 * (distance_km - 5.0)

    penalty = max(-10.0, penalty)

    multipliers = {
        "biospheric": 0.7,
        "altruistic": 0.9,
        "egoistic": 1.5,
        "hedonic": 1.8,
    }
    return penalty * multipliers.get(profile_type, 1.0)


def cycling_distance_penalty(distance_km: float, profile_type: str = "biospheric") -> float:
    """
    Penalty for long cycling distances.

    This is retained for compatibility with the current personalised router.
    """
    if distance_km <= 5.0:
        penalty = 0.0
    elif distance_km <= 8.0:
        penalty = -0.1 * (distance_km - 5.0)
    elif distance_km <= 12.0:
        penalty = -0.3 - 0.3 * (distance_km - 8.0)
    elif distance_km <= 15.0:
        penalty = -1.5 - 0.5 * (distance_km - 12.0)
    elif distance_km <= 20.0:
        penalty = -3.0 - 0.8 * (distance_km - 15.0)
    else:
        penalty = -7.0 - 1.0 * (distance_km - 20.0)

    penalty = max(-10.0, penalty)

    multipliers = {
        "biospheric": 0.6,
        "altruistic": 1.2,
        "egoistic": 1.3,
        "hedonic": 1.6,
    }
    return penalty * multipliers.get(profile_type, 1.0)

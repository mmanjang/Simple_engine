"""
value_model.py
──────────────
Defines the VALUE ATTRIBUTE MATRIX — how well each transport mode/strategy
satisfies each human need dimension from the DYCONET cognitive passport.

Scores are in the range [-1.0, +1.0]:
  +1.0  = this mode strongly satisfies this need
   0.0  = neutral
  -1.0  = this mode strongly conflicts with this need

The 11 need dimensions match the cognitive passport profile.needs keys exactly:
  pro_env          — environmental concern
  physical         — desire for physical exercise
  privacy          — preference for personal space / no crowding
  autonomy         — preference for self-directed travel
  cost             — sensitivity to monetary cost
  speed            — preference for fastest journey
  safety_accident  — concern about traffic/accident safety
  safety_crime     — concern about personal security
  comfort          — preference for comfortable, stress-free travel
  reliable         — preference for punctual, dependable travel
  health_infection — concern about infection risk in crowded spaces
"""

from dataclasses import dataclass
import math


# ----------------------------------------------
#  Value dimensions  (must match passport profile.needs keys exactly)
# ----------------------------------------------

VALUE_DIMENSIONS = [
    "pro_env",
    "physical",
    "privacy",
    "autonomy",
    "cost",
    "speed",
    "safety_accident",
    "safety_crime",
    "comfort",
    "reliable",
    "health_infection",
]


# ----------------------------------------------
#  Mode attribute matrix
#  Static prior scores per mode, used as a baseline.
#  Route-aware metric adjustments in personalised_router.py
#  are blended on top of these at METRIC_BLEND = 0.7.
# ----------------------------------------------

MODE_ATTRIBUTES = {

    "foot": {
        "pro_env":          1.0,
        "physical":         1.0,
        "privacy":          0.8,
        "autonomy":         0.8,
        "cost":             1.0,
        "speed":           -1.0,
        "safety_accident":  0.3,
        "safety_crime":    -0.2,
        "comfort":         -0.5,
        "reliable":         0.6,
        "health_infection": 0.9,
    },

    "bike": {
        "pro_env":          1.0,
        "physical":         0.9,
        "privacy":          0.8,
        "autonomy":         0.9,
        "cost":             0.8,
        "speed":            0.2,
        "safety_accident": -0.2,
        "safety_crime":     0.2,
        "comfort":         -0.2,
        "reliable":         0.7,
        "health_infection": 0.8,
    },

    "car": {
        "pro_env":         -1.0,
        "physical":        -1.0,
        "privacy":          1.0,
        "autonomy":         1.0,
        "cost":            -0.8,
        "speed":            0.9,
        "safety_accident":  0.2,
        "safety_crime":     0.9,
        "comfort":          0.9,
        "reliable":         0.8,
        "health_infection": 0.7,
    },

    "pt": {
        "pro_env":          0.8,
        "physical":         0.1,
        "privacy":         -0.8,
        "autonomy":        -0.8,
        "cost":             0.6,
        "speed":            0.3,
        "safety_accident":  0.8,
        "safety_crime":    -0.4,
        "comfort":          0.2,
        "reliable":         0.5,
        "health_infection": -0.8,
    },

    "bike_pt": {
        "pro_env":          0.9,
        "physical":         0.6,
        "privacy":         -0.2,
        "autonomy":         0.1,
        "cost":             0.7,
        "speed":            0.5,
        "safety_accident":  0.1,
        "safety_crime":    -0.1,
        "comfort":         -0.1,
        "reliable":         0.5,
        "health_infection": 0.0,
    },

    "car_pt": {
        "pro_env":         -0.2,
        "physical":        -0.5,
        "privacy":          0.2,
        "autonomy":         0.3,
        "cost":            -0.1,
        "speed":            0.8,
        "safety_accident":  0.5,
        "safety_crime":     0.4,
        "comfort":          0.7,
        "reliable":         0.7,
        "health_infection": -0.3,
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
    walk_d0_km: float = 2.35
    walk_k: float = 2.57
    bike_mu_km: float = 3.0
    bike_sigma_km: float = 1.5
    car_d0_km: float = 4.5
    car_k: float = 0.92
    pt_rise_k: float = 1.0
    pt_rise_d0_km: float = 2.5
    pt_decay_k: float = 0.25
    pt_decay_d0_km: float = 12.0
    transfer_alpha: float = 0.55
    transfer_distance_tolerance_base: float = 0.75
    transfer_distance_tolerance_slope: float = 0.18
    epsilon: float = 1e-6


DEFAULT_FEASIBILITY_PARAMS = DistanceFeasibilityParams()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def walk_distance_feasibility(distance_km, params=DEFAULT_FEASIBILITY_PARAMS):
    d = max(0.0, distance_km)
    return _clamp01(1.0 / (1.0 + math.exp(params.walk_k * (d - params.walk_d0_km))))


def bike_distance_feasibility(distance_km, params=DEFAULT_FEASIBILITY_PARAMS):
    d = max(0.0, distance_km)
    sigma = max(params.bike_sigma_km, params.epsilon)
    return _clamp01(math.exp(-((d - params.bike_mu_km) ** 2) / (2.0 * sigma ** 2)))


def car_distance_feasibility(distance_km, params=DEFAULT_FEASIBILITY_PARAMS):
    d = max(0.0, distance_km)
    return _clamp01(1.0 / (1.0 + math.exp(-params.car_k * (d - params.car_d0_km))))


def pt_distance_feasibility(distance_km, params=DEFAULT_FEASIBILITY_PARAMS):
    d = max(0.0, distance_km)
    rise  = 1.0 / (1.0 + math.exp(-params.pt_rise_k  * (d - params.pt_rise_d0_km)))
    decay = 1.0 / (1.0 + math.exp( params.pt_decay_k * (d - params.pt_decay_d0_km)))
    return _clamp01(rise * decay)


def transfer_feasibility(transfers, distance_km, params=DEFAULT_FEASIBILITY_PARAMS):
    n = max(0, int(transfers or 0))
    d = max(0.1, distance_km)
    tolerance = (params.transfer_distance_tolerance_base
                 + params.transfer_distance_tolerance_slope * d)
    return _clamp01(math.exp(-params.transfer_alpha * n / max(tolerance, params.epsilon)))


def mode_distance_feasibility(mode_key, distance_km, transfers=0,
                               params=DEFAULT_FEASIBILITY_PARAMS):
    mode_key = mode_key.lower()
    walk_f = walk_distance_feasibility(distance_km, params)
    bike_f = bike_distance_feasibility(distance_km, params)
    car_f  = car_distance_feasibility(distance_km, params)
    pt_f   = pt_distance_feasibility(distance_km, params)
    tf     = transfer_feasibility(transfers, distance_km, params)

    if mode_key == "foot":    return walk_f
    if mode_key == "bike":    return bike_f
    if mode_key == "car":     return car_f
    if mode_key == "pt":      return _clamp01(pt_f * tf)
    if mode_key == "bike_pt": return _clamp01((0.45 * bike_f + 0.55 * pt_f) * tf)
    if mode_key == "car_pt":
        gate = 1.0 / (1.0 + math.exp(-1.2 * (distance_km - 3.0)))
        return _clamp01((0.35 * car_f + 0.65 * pt_f) * tf * gate)
    return 0.0


def all_mode_distance_feasibilities(distance_km, transfers_by_mode=None,
                                     params=DEFAULT_FEASIBILITY_PARAMS):
    transfers_by_mode = transfers_by_mode or {}
    return {
        mode: mode_distance_feasibility(
            mode, distance_km,
            transfers=transfers_by_mode.get(mode, 0),
            params=params,
        )
        for mode in MODE_ATTRIBUTES.keys()
    }


def feasibility_log_term(feasibility, params=DEFAULT_FEASIBILITY_PARAMS):
    return math.log(max(params.epsilon, _clamp01(feasibility)))


# =============================================================================
# Route Metric Scoring Functions
# =============================================================================

def speed_score_from_duration(duration_s: float, reference_s: float = 1800) -> float:
    """Actual travel time → speed score [-1, +1]. Reference = 30 min."""
    if reference_s <= 0:
        return 0.0
    return max(-1.0, min(1.0, 1.0 - duration_s / reference_s))


def cost_score_from_mode(mode: str, distance_m: float) -> float:
    """Monetary cost proxy → score [-1, +1]. +1 = free, -1 = expensive."""
    distance_km = distance_m / 1000
    if mode == "foot":    return 1.0
    if mode == "bike":    return 0.9
    if mode == "pt":      return 0.5
    if mode == "bike_pt": return 0.6
    if mode == "car":
        return -min(1.0, distance_km * 0.30 / 10)
    if mode == "car_pt":
        return -min(1.0, distance_km * 0.15 / 10) * 0.5
    return 0.0


def comfort_score_from_transfers(transfers: int) -> float:
    """More transfers = less comfortable."""
    if transfers <= 0: return  0.5
    if transfers == 1: return  0.0
    return max(-1.0, -0.3 * transfers)


def walking_distance_penalty(distance_km: float,
                              profile_type: str = "biospheric") -> float:
    """Penalty for long walking distances, scaled by agent profile type."""
    if   distance_km <= 1.0: penalty = 0.0
    elif distance_km <= 2.0: penalty = -0.3  * (distance_km - 1.0)
    elif distance_km <= 3.0: penalty = -0.3  - 0.6 * (distance_km - 2.0)
    elif distance_km <= 5.0: penalty = -0.9  - 0.6 * (distance_km - 3.0)
    else:                    penalty = -2.1  - 2.0 * (distance_km - 5.0)
    penalty = max(-10.0, penalty)
    multipliers = {"biospheric": 0.7, "altruistic": 0.9,
                   "egoistic": 1.5,   "hedonic": 1.8}
    return penalty * multipliers.get(profile_type, 1.0)


def cycling_distance_penalty(distance_km: float,
                              profile_type: str = "biospheric") -> float:
    """Penalty for long cycling distances, scaled by agent profile type."""
    if   distance_km <=  5.0: penalty = 0.0
    elif distance_km <=  8.0: penalty = -0.1 * (distance_km -  5.0)
    elif distance_km <= 12.0: penalty = -0.3 - 0.3 * (distance_km -  8.0)
    elif distance_km <= 15.0: penalty = -1.5 - 0.5 * (distance_km - 12.0)
    elif distance_km <= 20.0: penalty = -3.0 - 0.8 * (distance_km - 15.0)
    else:                     penalty = -7.0 - 1.0 * (distance_km - 20.0)
    penalty = max(-10.0, penalty)
    multipliers = {"biospheric": 0.6, "altruistic": 1.2,
                   "egoistic": 1.3,   "hedonic": 1.6}
    return penalty * multipliers.get(profile_type, 1.0)
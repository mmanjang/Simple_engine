"""
wayfinder_app.py
----------------
Campus Wayfinder — standalone Flask app on port 5001.
Serves the wayfinder UI and provides a unified routing API
that auto-selects transport modes based on trip distance.

Endpoints:
    GET /                   — Wayfinder UI
    GET /api/wayfinder      — Unified routing endpoint (distance-adaptive)
    GET /api/route-simple   — Single-mode route (foot | bike | car)
    GET /api/route-pt       — Public transport route
    GET /api/health         — Health check
"""

import math
import os
import sys
from datetime import datetime, timezone

import requests as req_lib
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from graphhopper_client import GraphHopperClient
from gtfs_departures import GTFSDepartures


# -------------------------
#  CONFIG
# -------------------------

GRAPHHOPPER_HOST = os.getenv("GRAPHHOPPER_HOST", "http://localhost:8989")
GTFS_ZIP_PATH    = os.getenv("GTFS_ZIP_PATH",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "..", "data", "magdeburg_gtfs.zip"))
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))

gh_client    = GraphHopperClient(base_url=GRAPHHOPPER_HOST)
gtfs_board   = GTFSDepartures(GTFS_ZIP_PATH)

app = Flask(__name__)
CORS(app)

# Load GTFS on startup
try:
    gtfs_board.load()
    print("✅ GTFS departure board loaded")
except Exception as e:
    print(f"⚠️  GTFS load failed: {e}")


# -------------------------
#  UI
# -------------------------

@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "wayfinder.html"))


# -------------------------
#  DISTANCE HELPERS
# -------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    """Crow-flies distance in metres between two coordinates."""
    R     = 6_371_000
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a     = (math.sin(d_lat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def adaptive_modes(dist_m: float) -> list[dict]:
    """
    Return list of {mode, count} based on crow-flies distance.

    Campus-optimised thresholds:
      0–400 m    : 3 walk                           (just walk it)
      400–800 m  : 2 walk + 1 bike                  (walk or scooter)
      800–1500 m : 2 walk + 2 bike + 1 PT           (all short options)
      1500–4000 m: 1 walk + 2 bike + 1 PT + 1 car  (full campus range)
      4000 m+    : 1 bike + 1 PT + 1 car            (too far to walk)
    """
    if dist_m <= 400:
        return [{"mode": "foot", "count": 3}]
    elif dist_m <= 800:
        return [{"mode": "foot", "count": 2},
                {"mode": "bike", "count": 1}]
    elif dist_m <= 1500:
        return [{"mode": "foot", "count": 2},
                {"mode": "bike", "count": 2},
                {"mode": "pt",   "count": 1}]
    elif dist_m <= 4000:
        return [{"mode": "foot", "count": 1},
                {"mode": "bike", "count": 2},
                {"mode": "pt",   "count": 1},
                {"mode": "car",  "count": 1}]
    else:
        return [{"mode": "bike", "count": 1},
                {"mode": "pt",   "count": 1},
                {"mode": "car",  "count": 1}]


def distance_category(dist_m: float) -> dict:
    """Return display metadata for the distance band."""
    if dist_m <= 400:
        return {"category": "very_short", "label": "Very short · < 400 m",  "color": "#16a34a"}
    elif dist_m <= 800:
        return {"category": "short",      "label": "Short · < 800 m",       "color": "#0891b2"}
    elif dist_m <= 1500:
        return {"category": "medium",     "label": "Medium · < 1.5 km",     "color": "#d97706"}
    elif dist_m <= 4000:
        return {"category": "campus",     "label": "Campus range · < 4 km", "color": "#7c3aed"}
    else:
        return {"category": "long",       "label": "Long distance",         "color": "#6366f1"}


# -------------------------
#  ROUTE FETCHERS
# -------------------------

def fetch_gh_route(from_lat, from_lon, to_lat, to_lon, mode, count=1):
    """
    Call GraphHopper directly with instructions=true.
    Returns list of route dicts (one per alternative).
    """
    params = {
        "point":          [f"{from_lat},{from_lon}", f"{to_lat},{to_lon}"],
        "profile":        mode,
        "instructions":   "true",
        "points_encoded": "false",
    }
    if count > 1:
        params["algorithm"]                       = "alternative_route"
        params["alternative_route.max_paths"]     = count

    resp = req_lib.get(f"{GRAPHHOPPER_HOST}/route", params=params, timeout=15)
    resp.raise_for_status()
    paths = resp.json().get("paths", [])

    results = []
    for alt_idx, path in enumerate(paths[:count]):
        steps = [
            {
                "text":     s.get("text", ""),
                "distance": s.get("distance", 0),
                "sign":     str(s.get("sign", "")),
            }
            for s in path.get("instructions", [])
            if s.get("text")
        ]
        dist_m = path.get("distance", 0)
        results.append({
            "mode":       mode,
            "alt_index":  alt_idx,
            "duration_s": path.get("time", 0) / 1000,
            "distance_m": dist_m,
            "geometry":   path.get("points", {}),
            "steps":      steps,
            "transfers":  0,
            "walking_m":  dist_m if mode == "foot" else 0,
            "summary":    f"{round(dist_m / 1000, 1)} km",
        })
    return results


def fetch_pt_route(from_lat, from_lon, to_lat, to_lon):
    """
    Fetch PT route via the GraphHopperClient.
    Returns list with one route dict, or empty list on failure.
    """
    try:
        dep    = datetime.now(timezone.utc)
        routes = gh_client.route_pt(
            from_lat, from_lon, to_lat, to_lon,
            departure_time=dep, limit_solutions=1,
        )
        if not routes:
            return []

        r         = routes[0]
        legs      = getattr(r, "legs", None) or []
        walking_m = sum(
            getattr(l, "distance_m", 0) for l in legs
            if getattr(l, "mode", "") in ("walk", "foot")
        )

        steps = []
        for leg in legs:
            mode_name = getattr(leg, "mode", "unknown")
            dur_min   = round(getattr(leg, "duration_s", 0) / 60)
            dist_m    = getattr(leg, "distance_m", 0)
            if mode_name in ("walk", "foot"):
                steps.append({
                    "text":     f"Walk {round(dist_m)} m",
                    "distance": dist_m,
                    "sign":     "🚶",
                })
            else:
                rid      = getattr(leg, "route_id",      "") or ""
                headsign = getattr(leg, "trip_headsign", "") or ""
                label    = f"{rid} {headsign}".strip() or "Transit"
                steps.append({
                    "text":     f"Board {label} ({dur_min} min)",
                    "distance": dist_m,
                    "sign":     "🚌",
                })

        return [{
            "mode":       "pt",
            "alt_index":  0,
            "duration_s": r.duration_s,
            "distance_m": r.distance_m,
            "geometry":   r.geometry or r.points or {},
            "steps":      steps,
            "transfers":  getattr(r, "transfers", 0) or 0,
            "walking_m":  walking_m,
            "summary":    f"{getattr(r, 'transfers', 0) or 0} transfer(s)",
        }]
    except Exception:
        return []


# -------------------------
#  UNIFIED WAYFINDER API
# -------------------------

@app.route("/api/wayfinder", methods=["GET"])
def wayfinder_api():
    """
    Unified campus wayfinder endpoint.
    Computes crow-flies distance, selects appropriate modes,
    fetches all routes in parallel, and returns a single response.

    Params:
        from_lat, from_lon  — origin
        to_lat,   to_lon    — destination

    Response:
        {
          distance_m, distance_category, badge, color,
          modes_used, routes: [ { mode, alt_index, duration_s,
            distance_m, geometry, steps, transfers, walking_m, summary } ]
        }
    """
    try:
        from_lat = float(request.args["from_lat"])
        from_lon = float(request.args["from_lon"])
        to_lat   = float(request.args["to_lat"])
        to_lon   = float(request.args["to_lon"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing parameter: {e}"}), 400

    try:
        dist_m = haversine_m(from_lat, from_lon, to_lat, to_lon)
        modes  = adaptive_modes(dist_m)
        cat    = distance_category(dist_m)

        all_routes = []
        for m in modes:
            try:
                if m["mode"] == "pt":
                    all_routes.extend(fetch_pt_route(from_lat, from_lon, to_lat, to_lon))
                else:
                    all_routes.extend(fetch_gh_route(
                        from_lat, from_lon, to_lat, to_lon,
                        m["mode"], m["count"],
                    ))
            except Exception as e:
                app.logger.warning(f"Mode {m['mode']} failed: {e}")

        mode_order = {"foot": 0, "bike": 1, "pt": 2, "car": 3}
        all_routes.sort(key=lambda r: (mode_order.get(r["mode"], 9), r["duration_s"]))

        return jsonify({
            "distance_m":        round(dist_m),
            "distance_category": cat["category"],
            "badge":             cat["label"],
            "color":             cat["color"],
            "modes_used":        [m["mode"] for m in modes],
            "routes":            all_routes,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
#  SIMPLE MODE ROUTE
# -------------------------

@app.route("/api/route-simple", methods=["GET"])
def route_simple():
    """
    Single-mode route for foot | bike | car.
    Returns up to `count` alternative routes with turn-by-turn steps.

    Params: from_lat, from_lon, to_lat, to_lon, mode, count (default 1)
    """
    try:
        from_lat = float(request.args["from_lat"])
        from_lon = float(request.args["from_lon"])
        to_lat   = float(request.args["to_lat"])
        to_lon   = float(request.args["to_lon"])
        mode     = request.args.get("mode", "foot")
        count    = min(int(request.args.get("count", 1)), 3)
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing parameter: {e}"}), 400

    if mode not in ("foot", "bike", "car"):
        return jsonify({"error": f"Unknown mode: {mode}"}), 400

    try:
        results = fetch_gh_route(from_lat, from_lon, to_lat, to_lon, mode, count)
        if not results:
            return jsonify({"error": "No route found"}), 404
        return jsonify(results if count > 1 else results[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
#  PT ROUTE
# -------------------------

@app.route("/api/route-pt", methods=["GET"])
def route_pt():
    """
    Public transport route.

    Params: from_lat, from_lon, to_lat, to_lon
    """
    try:
        from_lat = float(request.args["from_lat"])
        from_lon = float(request.args["from_lon"])
        to_lat   = float(request.args["to_lat"])
        to_lon   = float(request.args["to_lon"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing parameter: {e}"}), 400

    results = fetch_pt_route(from_lat, from_lon, to_lat, to_lon)
    if not results:
        return jsonify({"error": "No PT route found"}), 404
    return jsonify(results[0])


# -------------------------
#  DEPARTURE BOARD
# -------------------------

@app.route("/api/departures", methods=["GET"])
def departures():
    """
    Next tram departures from the two university stops.

    Params (optional):
        limit   — max departures per stop (default 6)
        window  — look-ahead in minutes (default 90)

    Response:
        {
          "generated_at": "14:23:01",
          "stops": {
            "universitaet": {
              "label": "Magdeburg, Universität",
              "icon":  "🎓",
              "departures": [
                { "line": "6", "direction": "Herrenkrug",
                  "time": "14:28", "minutes": 5 },
                ...
              ]
            },
            "bibliothek": { ... }
          }
        }
    """
    try:
        limit  = int(request.args.get("limit",  6))
        window = int(request.args.get("window", 90))
        data   = gtfs_board.next_departures(limit=limit, window_minutes=window)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
#  HEALTH CHECK
# -------------------------

@app.route("/api/health")
def health():
    """Check GraphHopper connectivity."""
    try:
        resp = req_lib.get(f"{GRAPHHOPPER_HOST}/health", timeout=5)
        alive = resp.ok
    except Exception:
        alive = False
    return jsonify({
        "status":      "ok" if alive else "error",
        "graphhopper": "running" if alive else "not running",
    })


# -------------------------
#  ENTRY POINT
# -------------------------

if __name__ == "__main__":
    port = int(os.getenv("WAYFINDER_PORT", 5001))
    print(f"\n🗺️  Campus Wayfinder running on http://0.0.0.0:{port}")
    print(f"   GraphHopper: {GRAPHHOPPER_HOST}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
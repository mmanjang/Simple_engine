"""
Flask API server for VERA — Value-Aware Routing Assistant
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import Agent
from graphhopper_client import GraphHopperClient
from personalised_router import PersonalisedRouter

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

GRAPHHOPPER_HOST = os.getenv("GRAPHHOPPER_HOST", "http://localhost:8989")
gh_client = GraphHopperClient(base_url=GRAPHHOPPER_HOST)


def load_pois():
    try:
        with open('example_pois.json', 'r') as f:
            return json.load(f)
    except Exception:
        return []


def load_agent(agent_type):
    agent_files = {
        'biospheric': 'agents/agent_biospheric.json',
        'altruistic':  'agents/agent_altruistic.json',
        'egoistic':    'agents/agent_egoistic.json',
        'hedonic':     'agents/agent_hedonic.json',
    }
    with open(agent_files[agent_type], 'r') as f:
        return Agent.from_dict(json.load(f))


def build_custom_agent(needs: dict, beliefs: dict | None = None) -> Agent:
    # Clamp to [0,1] but do NOT L1-normalise — normalisation compresses the
    # differences between profiles and makes all rankings identical.
    # PersonalisedRouter uses the raw weights directly in the dot-product.
    weights = {k: max(0.0, min(1.0, float(v))) for k, v in needs.items()}
    resolved_beliefs = {
        "owns_bike":     bool((beliefs or {}).get("owns_bike",     True)),
        "owns_car":      bool((beliefs or {}).get("owns_car",      True)),
        "has_pt_access": bool((beliefs or {}).get("has_pt_access", True)),
    }
    agent_dict = {
        "id": "custom",
        "value_weights": weights,
        "beliefs": resolved_beliefs,
    }
    return Agent.from_dict(agent_dict)


@app.route('/')
def index():
    from flask import make_response
    resp = make_response(send_file('interface.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/demo')
def demo():
    from flask import make_response
    resp = make_response(send_file('vera_mobile.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    # Allow tile CDNs and Nominatim — without this browsers block the external images
    resp.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "img-src * data: blob:; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://unpkg.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self' "
        "https://nominatim.openstreetmap.org "
        "http://localhost:8989 http://localhost:5050; "
        "worker-src blob:; "
        "frame-src 'none';"
    )
    return resp


@app.route('/api/route', methods=['POST'])
def calculate_route():
    try:
        data        = request.json
        origin      = data['origin']
        destination = data['destination']
        pois_enabled = data.get('pois', False)

        if 'needs' in data and data['needs']:
            agent = build_custom_agent(data['needs'], data.get('beliefs'))
        else:
            agent = load_agent(data.get('agent', 'biospheric'))

        pois = load_pois() if pois_enabled else []

        router = PersonalisedRouter(
            gh_client,
            pois=pois if pois_enabled else None,
            poi_proximity_m=100,
        )

        from datetime import datetime, timezone
        departure = datetime(2025, 11, 15, 9, 0, tzinfo=timezone.utc)

        results = router.route(
            agent=agent,
            from_lat=origin[0], from_lon=origin[1],
            to_lat=destination[0], to_lon=destination[1],
            departure=departure.isoformat(),
            max_walk_m=1500,
        )

        routes = []
        for result in results:
            route_geometry = getattr(result.route, 'geometry', None)

            # ── Leg breakdown ─────────────────────────────────────────────
            legs_data = []
            for leg in result.route.legs:
                leg_info = {
                    'type':       getattr(leg, 'type', getattr(leg, 'mode', 'unknown')),
                    'mode':       getattr(leg, 'mode', getattr(leg, 'type', 'unknown')),
                    'distance_m': getattr(leg, 'distance_m', 0.0),
                    'duration_s': getattr(leg, 'duration_s', 0.0),
                    'geometry':   getattr(leg, 'geometry', None),
                }
                for attr in ('from_name', 'to_name', 'route_id', 'trip_headsign',
                             'departure_time', 'arrival_time', 'from_stop',
                             'to_stop', 'num_stops', 'stops'):
                    if hasattr(leg, attr):
                        leg_info[attr] = getattr(leg, attr)
                legs_data.append(leg_info)

            # ── Dimension scores — defensive: work with or without new attrs ──
            dim_scores_raw  = []
            top_matching    = []
            top_conflicting = []

            raw_dims = getattr(result, 'dimension_scores', None)
            if raw_dims:
                for d in raw_dims:
                    dim_scores_raw.append({
                        'dimension':    getattr(d, 'dimension', ''),
                        'weight':       round(getattr(d, 'agent_weight', 0), 3),
                        'attribute':    round(getattr(d, 'blended_attribute', 0), 3),
                        'contribution': round(getattr(d, 'contribution', 0), 3),
                    })

            for d in getattr(result, 'top_matching_values', []):
                top_matching.append({
                    'dimension':    getattr(d, 'dimension', ''),
                    'contribution': round(getattr(d, 'contribution', 0), 3),
                })

            for d in getattr(result, 'top_conflicting_values', []):
                top_conflicting.append({
                    'dimension':    getattr(d, 'dimension', ''),
                    'contribution': round(getattr(d, 'contribution', 0), 3),
                })

            # ── Geometry ──────────────────────────────────────────────────
            geometry_geojson = None
            if route_geometry and 'coordinates' in route_geometry:
                geometry_geojson = {
                    'type': 'LineString',
                    'coordinates': route_geometry['coordinates'],
                }

            routes.append({
                'rank':             result.rank,
                'mode':             result.mode_key,
                'mode_label':       result.mode_label,
                'score':            result.utility_score,   # raw; normalised below
                'feasibility':      round(getattr(result, 'feasibility_score', 1.0), 3),
                'time_min':         result.route.duration_min,
                'distance_km':      result.route.distance_km,
                'available':        result.available,
                'poi_boost':        round(getattr(result, 'poi_boost', 1.0), 2),
                'matched_pois':     getattr(result, 'matched_pois', []),
                'geometry':         geometry_geojson,
                'legs':             legs_data,
                'dimension_scores': dim_scores_raw,
                'top_matching':     top_matching,
                'top_conflicting':  top_conflicting,
            })

        # ── Normalise scores to [0, 100] across ALL routes (available or not) ──
        # Unavailable routes are scored honestly in the same scale — they may
        # rank higher than available ones if the agent's values prefer them,
        # which is intentional: it shows "this would have been best for you."
        raw_scores = [r['score'] for r in routes]
        s_min, s_max = min(raw_scores), max(raw_scores)
        score_range = s_max - s_min if s_max != s_min else 1.0
        for r in routes:
            r['raw_score'] = round(r['score'], 3)
            r['score']     = round((r['score'] - s_min) / score_range * 100, 1)

        return jsonify(routes)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR in /api/route: {e}\n{tb}")   # always visible in docker logs
        return jsonify({
            'error':     str(e),
            'type':      type(e).__name__,
            'traceback': tb,
        }), 500


@app.route('/api/agents', methods=['GET'])
def get_agents():
    agents = {}
    for agent_type in ['biospheric', 'altruistic', 'egoistic', 'hedonic']:
        agent = load_agent(agent_type)
        top_values = sorted(
            agent.value_weights.items(), key=lambda x: x[1], reverse=True
        )[:5]
        agents[agent_type] = {
            'id':         agent.id,
            'top_values': {k: round(v, 2) for k, v in top_values},
            'beliefs':    agent.beliefs,
        }
    return jsonify(agents)


@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        alive = gh_client.is_alive()
        return jsonify({
            'status':      'ok' if alive else 'error',
            'graphhopper': 'running' if alive else 'not running',
            'message':     'All systems operational' if alive else 'GraphHopper not responding',
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  VERA — Value-Aware Routing Assistant")
    print("=" * 60)
    print(f"\n  Main UI  : http://localhost:5050")
    print(f"  Demo UI  : http://localhost:5050/demo")
    print(f"  API      : http://localhost:5050/api/route")
    print(f"  Health   : http://localhost:5050/api/health")
    print(f"  GraphHopper: {GRAPHHOPPER_HOST}")
    print("=" * 60 + "\n")
    try:
        if gh_client.is_alive():
            print("✅  GraphHopper connected\n")
        else:
            print("❌  GraphHopper not responding\n")
    except Exception:
        print("❌  Could not connect to GraphHopper\n")

    app.run(debug=True, port=5050, host='0.0.0.0')
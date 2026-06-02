"""
Flask API server for Value-Based Routing Engine
Connects the web interface to the Python routing engine
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sys, os


# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
# from study_db import get_connection, init_database, seed_scenarios

from agent import Agent
from graphhopper_client import GraphHopperClient
from personalised_router import PersonalisedRouter
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Initialize GraphHopper client
GRAPHOPPER_HOST = os.getenv("GRAPHOPPER_HOST", "http://localhost:8080")
gh_client = GraphHopperClient(base_url=GRAPHOPPER_HOST)

# Load POIs
def load_pois():
    try:
        with open('example_pois.json', 'r') as f:
            return json.load(f)
    except:
        return []

# Load agent profiles
def load_agent(agent_type):
    """Load agent profile from JSON file"""
    agent_files = {
        'biospheric': 'agents/agent_biospheric.json',
        'altruistic': 'agents/agent_altruistic.json',
        'egoistic': 'agents/agent_egoistic.json',
        'hedonic': 'agents/agent_hedonic.json'
    }
    
    with open(agent_files[agent_type], 'r') as f:
        agent_data = json.load(f)
    
    return Agent.from_dict(agent_data)


def extract_leg_geometry(leg, route_points):
    """
    Extract geometry for a specific leg from the overall route geometry.
    
    For walk legs: Extract segment from route.points
    For PT legs: Extract segment between stops from route.points
    """
    # route_points should be in format: {"type": "LineString", "coordinates": [[lon, lat], ...]}
    if not route_points or not route_points.get('coordinates'):
        return None
    
    coords = route_points['coordinates']  # [[lon, lat], ...]
    
    # For now, return the full geometry
    # In a more sophisticated version, you'd slice this based on leg start/end
    return {
        "type": "LineString",
        "coordinates": coords
    }


@app.route('/')
def index():
    """Serve the main interface"""
    return send_file('interface.html')

@app.route('/api/route', methods=['POST'])
def calculate_route():
    """
    Calculate value-based routes
    
    Request JSON:
    {
        "origin": [lat, lon],
        "destination": [lat, lon],
        "agent": "biospheric|altruistic|egoistic|hedonic",
        "pois": true|false
    }
    
    Response JSON:
    [
        {
            "rank": 1,
            "mode": "bike",
            "mode_label": "🚴 Bike",
            "score": 94.2,
            "time_min": 28,
            "distance_km": 8.0,
            "available": true,
            "poi_boost": 1.2,
            "matched_pois": [...],
            "geometry": {...},  # GeoJSON for overall route
            "legs": [...]  # Detailed leg-by-leg geometry
        },
        ...
    ]
    """
    try:
        data = request.json
        
        # Extract parameters
        origin = data['origin']  # [lat, lon]
        destination = data['destination']
        agent_type = data.get('agent', 'biospheric')
        pois_enabled = data.get('pois', True)
        
        # Load agent
        agent = load_agent(agent_type)
        
        # Load POIs if enabled
        pois = load_pois() if pois_enabled else []
        
        # Initialize router
        router = PersonalisedRouter(
            gh_client, 
            pois=pois if pois_enabled else None,
            poi_proximity_m=100
        )
        
        # Calculate routes
        from datetime import datetime, timezone
        departure = datetime(2025, 11, 15, 9, 0, tzinfo=timezone.utc)
        
        results = router.route(
            agent=agent,
            from_lat=origin[0],
            from_lon=origin[1],
            to_lat=destination[0],
            to_lon=destination[1],
            departure=departure.isoformat(),
            max_walk_m=1500
        )
        
        # Format response
        routes = []
        for result in results:
            # Get overall route geometry
            route_geometry = getattr(result.route, 'geometry', None)
            
            # Extract leg-by-leg information for visualization
            legs_data = []
            for leg in result.route.legs:
                leg_info = {
                    'type': getattr(leg, 'type', getattr(leg, 'mode', 'unknown')),
                    'mode': getattr(leg, 'mode', getattr(leg, 'type', 'unknown')),
                    'distance_m': getattr(leg, 'distance_m', 0.0),
                    'duration_s': getattr(leg, 'duration_s', 0.0),
                    'geometry': getattr(leg, 'geometry', None),   # <-- ADD THIS
                }

                if hasattr(leg, 'from_name'):
                    leg_info['from_name'] = leg.from_name
                if hasattr(leg, 'to_name'):
                    leg_info['to_name'] = leg.to_name

                if hasattr(leg, 'route_id'):
                    leg_info['route_id'] = getattr(leg, 'route_id', '')
                    leg_info['trip_headsign'] = getattr(leg, 'trip_headsign', '')
                    leg_info['departure_time'] = getattr(leg, 'departure_time', None)
                    leg_info['arrival_time'] = getattr(leg, 'arrival_time', None)
                    leg_info['from_stop'] = getattr(leg, 'from_stop', '')
                    leg_info['to_stop'] = getattr(leg, 'to_stop', '')
                    leg_info['num_stops'] = getattr(leg, 'num_stops', 0)
                    leg_info['stops'] = getattr(leg, 'stops', [])  # keep stops for markers only

                legs_data.append(leg_info)
            
            # Convert geometry to proper GeoJSON format
            geometry_geojson = None
            if route_geometry and 'coordinates' in route_geometry:
                geometry_geojson = {
                    "type": "LineString",
                    "coordinates": route_geometry['coordinates']
                }
            
            routes.append({
                'rank': result.rank,
                'mode': result.mode_key,
                'mode_label': result.mode_label,
                'score': round(result.utility_score, 1),
                'time_min': result.route.duration_min,
                'distance_km': result.route.distance_km,
                'available': result.available,
                'poi_boost': round(result.poi_boost, 2),
                'matched_pois': result.matched_pois,
                'geometry': geometry_geojson,  # Properly formatted GeoJSON
                'legs': legs_data
            })
        
        return jsonify(routes)
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get available agent profiles with their values"""
    agents = {}
    
    for agent_type in ['biospheric', 'altruistic', 'egoistic', 'hedonic']:
        agent = load_agent(agent_type)
        
        # Get top values
        top_values = sorted(
            agent.value_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        agents[agent_type] = {
            'id': agent.id,
            'top_values': {k: round(v, 2) for k, v in top_values},
            'beliefs': agent.beliefs
        }
    
    return jsonify(agents)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if GraphHopper server is running"""
    try:
        is_alive = gh_client.is_alive()
        return jsonify({
            'status': 'ok' if is_alive else 'error',
            'graphhopper': 'running' if is_alive else 'not running',
            'message': 'All systems operational' if is_alive else 'GraphHopper server not responding'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🗺️  Value-Based Routing Engine - Web Interface")
    print("="*60)
    print("\n📍 Starting server...")
    print(f"   Interface: http://localhost:5000")
    print(f"   API: http://localhost:5000/api/route")
    print(f"\n⚠️  Make sure GraphHopper is running on {GRAPHOPPER_HOST}")
    print("="*60 + "\n")
    
    # Check GraphHopper connection
    try:
        if gh_client.is_alive():
            print("✅ GraphHopper server connected\n")
        else:
            print("❌ WARNING: GraphHopper server not responding")
            print("   Start it with: java -Xmx4g -jar graphhopper/graphhopper-web-10.0.jar server graphhopper/config.yml\n")
    except:
        print("❌ WARNING: Could not connect to GraphHopper")
        print("   Start it with: java -Xmx4g -jar graphhopper/graphhopper-web-10.0.jar server graphhopper/config.yml\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
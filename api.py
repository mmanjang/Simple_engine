
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from routing_service import RoutingService, RoutingServiceError


GRAPHOPPER_HOST = os.getenv("GRAPHOPPER_HOST", "http://localhost:8080")
DEFAULT_POIS_FILE = os.getenv("POIS_FILE")
DEFAULT_POI_PROXIMITY_M = int(os.getenv("POI_PROXIMITY_M", "100"))

app = FastAPI(
    title="Personalised Multimodal Routing API",
    version="0.1.0",
    description="Ranks multimodal routes using a cognitive passport, start/stop coordinates, and departure datetime.",
)


class Coordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class RouteRequest(BaseModel):
    cognitive_passport: Dict[str, Any] = Field(
        ...,
        description="Agent profile from the app. Expected keys include id, values, and beliefs.",
    )
    start: Coordinate
    stop: Coordinate
    datetime: Optional[str] = Field(
        None,
        description="Departure datetime. Use an ISO string accepted by the routing backend, or null for now.",
    )
    max_walk_m: int = Field(500, ge=0, description="Maximum walking distance in meters")
    include_unavailable: bool = Field(True, description="Return unavailable modes with reasons")


class HealthResponse(BaseModel):
    status: str
    graphhopper_host: str


def load_pois(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("POIs file must contain a JSON list.")
    return data


def get_routing_service(check_graphhopper: bool = True) -> RoutingService:
    try:
        return RoutingService(
            graphhopper_host=GRAPHOPPER_HOST,
            pois=load_pois(DEFAULT_POIS_FILE),
            poi_proximity_m=DEFAULT_POI_PROXIMITY_M,
            check_graphhopper=check_graphhopper,
        )
    except RoutingServiceError:
        raise
    except Exception as exc:
        raise RoutingServiceError(str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_routing_service(check_graphhopper=True)
        return HealthResponse(status="ok", graphhopper_host=GRAPHOPPER_HOST)
    except RoutingServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ranked-routes")
def ranked_routes(request: RouteRequest) -> Dict[str, Any]:
    try:
        service = get_routing_service(check_graphhopper=True)
        return service.rank_routes(
            cognitive_passport=request.cognitive_passport,
            start=request.start.model_dump() if hasattr(request.start, "model_dump") else request.start.dict(),
            stop=request.stop.model_dump() if hasattr(request.stop, "model_dump") else request.stop.dict(),
            departure=request.datetime,
            max_walk_m=request.max_walk_m,
            include_unavailable=request.include_unavailable,
        )
    except RoutingServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Routing failed: {exc}") from exc

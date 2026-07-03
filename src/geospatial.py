"""Open geospatial lookup helpers for metric facade scaling."""

from __future__ import annotations

import json
import math
import re
import socket
import urllib.error
import urllib.parse
import urllib.request


OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
)
EARTH_RADIUS_M = 6_371_008.8


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lam = math.radians(lon2 - lon1)
    y = math.sin(d_lam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lam)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _bearing_delta_deg(a: float, b: float) -> float:
    """Smallest difference between two undirected edge bearings."""

    delta = abs((a - b + 180) % 360 - 180)
    return min(delta, abs(delta - 180))


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def _parse_height_m(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+", value.replace(",", "."))
    if not match:
        return None
    return float(match.group(0))


def _parse_levels(value: str | int | float | None) -> int | None:
    height = _parse_height_m(value)
    if height is None:
        return None
    return max(int(round(height)), 1)


def _edge_measurements(points: list[tuple[float, float]]) -> list[dict]:
    if len(points) < 2:
        return []
    closed = points
    if points[0] != points[-1]:
        closed = [*points, points[0]]

    edges = []
    for index, (start, end) in enumerate(zip(closed[:-1], closed[1:])):
        lat1, lon1 = start
        lat2, lon2 = end
        length = _haversine_m(lat1, lon1, lat2, lon2)
        if length <= 0.25:
            continue
        edges.append(
            {
                "index": index,
                "length_m": length,
                "bearing_deg": _bearing_deg(lat1, lon1, lat2, lon2),
                "start": [lat1, lon1],
                "end": [lat2, lon2],
            }
        )
    return edges


def _select_facade_edge(edges: list[dict], facade_bearing_deg: float | None = None) -> tuple[dict | None, str]:
    if not edges:
        return None, "no-footprint-edges"

    if facade_bearing_deg is not None:
        selected = min(
            edges,
            key=lambda edge: _bearing_delta_deg(float(edge["bearing_deg"]), facade_bearing_deg),
        )
        return selected, "osm-footprint-bearing-matched-edge"

    # A simple first-pass frontage prior. It is intentionally transparent and
    # recorded in the output so it can be replaced by camera-bearing matching.
    selected = max(edges, key=lambda edge: float(edge["length_m"]))
    return selected, "osm-footprint-longest-edge"


def _element_points(element: dict) -> list[tuple[float, float]]:
    geometry = element.get("geometry") or []
    points = []
    for point in geometry:
        if "lat" in point and "lon" in point:
            points.append((float(point["lat"]), float(point["lon"])))
    return points


def _building_candidates(payload: dict, target_lat: float, target_lon: float) -> list[dict]:
    candidates = []
    for element in payload.get("elements", []):
        points = _element_points(element)
        if len(points) < 3:
            continue
        centroid_lat, centroid_lon = _centroid(points)
        distance_m = _haversine_m(target_lat, target_lon, centroid_lat, centroid_lon)
        candidates.append(
            {
                "osm_id": element.get("id"),
                "osm_type": element.get("type"),
                "tags": element.get("tags", {}),
                "points": points,
                "centroid": [centroid_lat, centroid_lon],
                "distance_to_query_m": distance_m,
            }
        )
    candidates.sort(key=lambda item: item["distance_to_query_m"])
    return candidates


def _overpass_payload(query: str) -> tuple[dict, str]:
    """Request Overpass JSON, trying public mirrors if one endpoint rejects us."""

    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    headers = {
        "User-Agent": "BIPV_Project/0.1 academic-research",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }
    errors = []
    for url in OVERPASS_URLS:
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                return json.loads(response.read().decode("utf-8")), url
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:240]
            errors.append(f"{url}: HTTP {exc.code} {body}".strip())
        except urllib.error.URLError as exc:
            errors.append(f"{url}: {exc.reason}")
        except (TimeoutError, socket.timeout) as exc:
            errors.append(f"{url}: timeout {exc}")

    raise ValueError("All Overpass endpoints failed. " + " | ".join(errors))


def fetch_osm_building_reference(
    lat: float,
    lon: float,
    *,
    radius_m: float = 60.0,
    facade_bearing_deg: float | None = None,
    default_floor_height_m: float = 3.3,
) -> dict:
    """Fetch and measure the nearest OSM building footprint around a coordinate."""

    query = f"""
    [out:json][timeout:25];
    (
      way["building"](around:{float(radius_m)},{float(lat)},{float(lon)});
      relation["building"](around:{float(radius_m)},{float(lat)},{float(lon)});
    );
    out tags geom center;
    """
    payload, endpoint_url = _overpass_payload(query)

    candidates = _building_candidates(payload, lat, lon)
    if not candidates:
        raise ValueError(f"No OSM building footprint found within {radius_m:.0f} m.")

    building = candidates[0]
    edges = _edge_measurements(building["points"])
    facade_edge, width_method = _select_facade_edge(edges, facade_bearing_deg=facade_bearing_deg)
    if facade_edge is None:
        raise ValueError("OSM building footprint did not contain measurable edges.")

    tags = building["tags"]
    height_m = _parse_height_m(tags.get("height"))
    levels = _parse_levels(tags.get("building:levels") or tags.get("levels"))
    height_source = None
    if height_m is not None:
        height_source = "osm-height-tag"
    elif levels is not None:
        height_m = levels * default_floor_height_m
        height_source = "osm-building-levels"

    return {
        "source": "osm-overpass",
        "overpass_endpoint": endpoint_url,
        "osm_id": building["osm_id"],
        "osm_type": building["osm_type"],
        "distance_to_query_m": building["distance_to_query_m"],
        "facade_width_m": float(facade_edge["length_m"]),
        "facade_width_source": width_method,
        "facade_edge": facade_edge,
        "height_m": height_m,
        "height_source": height_source,
        "building_levels": levels,
        "tags": tags,
        "footprint_edge_count": len(edges),
        "footprint_centroid": building["centroid"],
    }

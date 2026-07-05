"""Open geospatial lookup helpers for metric facade scaling."""

from __future__ import annotations

import json
import math
import re
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from html import escape
from pathlib import Path


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


def _bbox_around(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    lat_delta = radius_m / 111_320
    lon_delta = radius_m / (111_320 * max(math.cos(math.radians(lat)), 0.15))
    return lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta


def _geojson_ring_points(ring: list) -> list[tuple[float, float]]:
    points = []
    for coord in ring or []:
        if len(coord) >= 2:
            lon, lat = coord[:2]
            points.append((float(lat), float(lon)))
    return points


def _geojson_polygon_points(geometry: dict) -> list[tuple[float, float]]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if geom_type == "Polygon" and coords:
        return _geojson_ring_points(coords[0])
    if geom_type == "MultiPolygon" and coords:
        largest = max(coords, key=lambda polygon: len(polygon[0]) if polygon else 0)
        return _geojson_ring_points(largest[0]) if largest else []
    return []


def _overture_features_from_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("{"):
        payload = json.loads(text)
        if payload.get("type") == "FeatureCollection":
            return payload.get("features", [])
        if payload.get("type") == "Feature":
            return [payload]

    features = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "Feature":
            features.append(payload)
    return features


def _run_overture_download(lat: float, lon: float, radius_m: float, path: Path) -> None:
    west, south, east, north = _bbox_around(lat, lon, radius_m)
    bbox = f"{west},{south},{east},{north}"
    commands = (
        [
            sys.executable,
            "-m",
            "overturemaps",
            "download",
            "--bbox",
            bbox,
            "-f",
            "geojson",
            "--type",
            "building",
            "-o",
            str(path),
        ],
        [
            "overturemaps",
            "download",
            "--bbox",
            bbox,
            "-f",
            "geojson",
            "--type",
            "building",
            "-o",
            str(path),
        ],
    )
    errors = []
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{command[0]}: {exc}")
            continue
        if completed.returncode == 0 and path.exists():
            return
        errors.append((completed.stderr or completed.stdout or "").strip())
    raise ValueError(
        "Overture Maps lookup failed. Install `overturemaps` and retry. "
        + " | ".join(error for error in errors if error)
    )


def fetch_overture_building_reference(
    lat: float,
    lon: float,
    *,
    radius_m: float = 60.0,
    facade_bearing_deg: float | None = None,
    default_floor_height_m: float = 3.3,
) -> dict:
    """Fetch and measure nearest Overture Maps building footprint."""

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "overture_buildings.geojson"
        _run_overture_download(lat, lon, radius_m, output_path)
        features = _overture_features_from_file(output_path)

    candidates = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        points = _geojson_polygon_points(geometry)
        if len(points) < 3:
            continue
        centroid_lat, centroid_lon = _centroid(points)
        candidates.append(
            {
                "feature": feature,
                "points": points,
                "centroid": [centroid_lat, centroid_lon],
                "distance_to_query_m": _haversine_m(lat, lon, centroid_lat, centroid_lon),
            }
        )
    candidates.sort(key=lambda item: item["distance_to_query_m"])
    if not candidates:
        raise ValueError(f"No Overture building footprint found within {radius_m:.0f} m.")

    building = candidates[0]
    properties = building["feature"].get("properties") or {}
    edges = _edge_measurements(building["points"])
    facade_edge, width_method = _select_facade_edge(edges, facade_bearing_deg=facade_bearing_deg)
    if facade_edge is None:
        raise ValueError("Overture building footprint did not contain measurable edges.")

    height_m = _parse_height_m(
        properties.get("height")
        or properties.get("building_height")
        or properties.get("min_height")
    )
    levels = _parse_levels(
        properties.get("num_floors")
        or properties.get("building_levels")
        or properties.get("levels")
    )
    height_source = None
    if height_m is not None:
        height_source = "overture-height"
    elif levels is not None:
        height_m = levels * default_floor_height_m
        height_source = "overture-building-levels"

    return {
        "source": "overture-maps",
        "overture_id": properties.get("id") or building["feature"].get("id"),
        "distance_to_query_m": building["distance_to_query_m"],
        "facade_width_m": float(facade_edge["length_m"]),
        "facade_width_source": f"overture-{width_method}",
        "facade_edge": facade_edge,
        "height_m": height_m,
        "height_source": height_source,
        "building_levels": levels,
        "tags": properties,
        "footprint_points": [[lat, lon] for lat, lon in building["points"]],
        "footprint_edges": edges,
        "footprint_edge_count": len(edges),
        "footprint_centroid": building["centroid"],
    }


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
      node["building"](around:{float(radius_m)},{float(lat)},{float(lon)});
      way["building"](around:{float(radius_m)},{float(lat)},{float(lon)});
      relation["building"](around:{float(radius_m)},{float(lat)},{float(lon)});
      way["building:part"](around:{float(radius_m)},{float(lat)},{float(lon)});
      relation["building:part"](around:{float(radius_m)},{float(lat)},{float(lon)});
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
        "footprint_points": [[lat, lon] for lat, lon in building["points"]],
        "footprint_edges": edges,
        "footprint_edge_count": len(edges),
        "footprint_centroid": building["centroid"],
    }


def fetch_geospatial_building_reference(
    lat: float,
    lon: float,
    *,
    radius_m: float = 60.0,
    facade_bearing_deg: float | None = None,
    default_floor_height_m: float = 3.3,
) -> dict:
    """Fetch building dimensions from Overture first, then OSM/Overpass."""

    errors = []
    try:
        return fetch_overture_building_reference(
            lat,
            lon,
            radius_m=radius_m,
            facade_bearing_deg=facade_bearing_deg,
            default_floor_height_m=default_floor_height_m,
        )
    except Exception as exc:
        errors.append(f"overture-maps: {type(exc).__name__}: {exc}")

    try:
        reference = fetch_osm_building_reference(
            lat,
            lon,
            radius_m=radius_m,
            facade_bearing_deg=facade_bearing_deg,
            default_floor_height_m=default_floor_height_m,
        )
        reference["fallback_errors"] = errors
        return reference
    except Exception as exc:
        errors.append(f"osm-overpass: {type(exc).__name__}: {exc}")

    raise ValueError("No geospatial building footprint found. " + " | ".join(errors))


def geospatial_reference_geojson(
    reference: dict,
    *,
    query_lat: float | None = None,
    query_lon: float | None = None,
) -> dict:
    """Build GeoJSON that verifies the selected building and facade edge."""

    features = []
    points = reference.get("footprint_points") or []
    if points:
        ring = [[float(lon), float(lat)] for lat, lon in points]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": "Selected building footprint",
                    "source": reference.get("source"),
                    "overture_id": reference.get("overture_id"),
                    "osm_id": reference.get("osm_id"),
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )

    for edge in reference.get("footprint_edges") or []:
        start = edge.get("start")
        end = edge.get("end")
        if not start or not end:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": f"Footprint edge {edge.get('index')}",
                    "length_m": edge.get("length_m"),
                    "bearing_deg": edge.get("bearing_deg"),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(start[1]), float(start[0])],
                        [float(end[1]), float(end[0])],
                    ],
                },
            }
        )

    facade_edge = reference.get("facade_edge") or {}
    start = facade_edge.get("start")
    end = facade_edge.get("end")
    if start and end:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": "Selected facade edge",
                    "length_m": facade_edge.get("length_m"),
                    "bearing_deg": facade_edge.get("bearing_deg"),
                    "facade_width_source": reference.get("facade_width_source"),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(start[1]), float(start[0])],
                        [float(end[1]), float(end[0])],
                    ],
                },
            }
        )

    centroid = reference.get("footprint_centroid")
    if centroid:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": "Selected footprint centroid",
                    "distance_to_query_m": reference.get("distance_to_query_m"),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(centroid[1]), float(centroid[0])],
                },
            }
        )

    if query_lat is not None and query_lon is not None:
        features.append(
            {
                "type": "Feature",
                "properties": {"name": "Input address/coordinate point"},
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(query_lon), float(query_lat)],
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def save_geospatial_reference_geojson(
    reference: dict,
    output_path: str | Path,
    *,
    query_lat: float | None = None,
    query_lon: float | None = None,
) -> str:
    """Save a GeoJSON verification file for the selected building reference."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = geospatial_reference_geojson(
        reference,
        query_lat=query_lat,
        query_lon=query_lon,
    )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(output_path)


def save_geospatial_reference_map(
    reference: dict,
    output_path: str | Path,
    *,
    query_lat: float | None = None,
    query_lon: float | None = None,
) -> str:
    """Save a lightweight Leaflet HTML map for visual verification."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centroid = reference.get("footprint_centroid") or [query_lat, query_lon]
    center_lat = float(centroid[0] if centroid and centroid[0] is not None else 0)
    center_lon = float(centroid[1] if centroid and centroid[1] is not None else 0)
    geojson = geospatial_reference_geojson(
        reference,
        query_lat=query_lat,
        query_lon=query_lon,
    )
    title = escape(
        f"{reference.get('source', 'geospatial')} "
        f"{reference.get('overture_id') or reference.get('osm_id') or ''}"
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>BIPV Geospatial Reference Verification</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    .info {{
      position: absolute;
      z-index: 1000;
      left: 12px;
      top: 12px;
      max-width: 420px;
      padding: 10px 12px;
      background: white;
      border: 1px solid #bbb;
      border-radius: 4px;
      font: 14px/1.35 Arial, sans-serif;
    }}
  </style>
</head>
<body>
<div id="map"></div>
<div class="info">
  <strong>BIPV Geospatial Verification</strong><br>
  Source: {escape(str(reference.get('source')))}<br>
  Building ID: {title}<br>
  Selected facade width: {float(reference.get('facade_width_m', 0)):.2f} m<br>
  Selected edge source: {escape(str(reference.get('facade_width_source')))}<br>
  Distance from input point: {float(reference.get('distance_to_query_m', 0)):.2f} m
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map').setView([{center_lat}, {center_lon}], 19);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 22,
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

const data = {json.dumps(geojson)};
function style(feature) {{
  const name = feature.properties && feature.properties.name || '';
  if (name === 'Selected building footprint') {{
    return {{color: '#1f77b4', weight: 2, fillColor: '#1f77b4', fillOpacity: 0.18}};
  }}
  if (name === 'Selected facade edge') {{
    return {{color: '#e31a1c', weight: 6}};
  }}
  if (name.startsWith('Footprint edge')) {{
    return {{color: '#888', weight: 1, dashArray: '4 4'}};
  }}
  return {{color: '#333', weight: 2}};
}}
function pointToLayer(feature, latlng) {{
  const name = feature.properties && feature.properties.name || '';
  const color = name === 'Input address/coordinate point' ? '#ff7f00' : '#33a02c';
  return L.circleMarker(latlng, {{radius: 7, color, fillColor: color, fillOpacity: 0.9}});
}}
const layer = L.geoJSON(data, {{
  style,
  pointToLayer,
  onEachFeature: (feature, layer) => {{
    const props = feature.properties || {{}};
    layer.bindPopup(Object.entries(props).map(([k,v]) => `<strong>${{k}}</strong>: ${{v}}`).join('<br>'));
  }}
}}).addTo(map);
map.fitBounds(layer.getBounds(), {{padding: [30, 30]}});
</script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)

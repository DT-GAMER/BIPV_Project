"""Address geocoding helpers for optional geospatial scaling."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


GOOGLE_GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def geocode_google(address: str, api_key: str | None = None) -> dict:
    """Resolve an address to latitude/longitude using Google Geocoding API."""

    key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    if not key:
        raise ValueError(
            "Google Maps API key is required for geocoding. Set "
            "GOOGLE_MAPS_API_KEY or pass google_maps_api_key in AnalysisConfig."
        )

    query = urllib.parse.urlencode({"address": address, "key": key})
    url = f"{GOOGLE_GEOCODING_URL}?{query}"
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    status = payload.get("status")
    if status != "OK" or not payload.get("results"):
        message = payload.get("error_message") or f"Google geocoding failed: {status}"
        raise ValueError(message)

    result = payload["results"][0]
    location = result["geometry"]["location"]
    return {
        "lat": float(location["lat"]),
        "lon": float(location["lng"]),
        "formatted_address": result.get("formatted_address"),
        "place_id": result.get("place_id"),
        "source": "google-geocoding",
    }


def resolve_coordinates(
    *,
    address: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    google_maps_api_key: str | None = None,
) -> dict | None:
    """Return coordinates from explicit lat/lon or a geocoded address."""

    if latitude is not None and longitude is not None:
        return {
            "lat": float(latitude),
            "lon": float(longitude),
            "source": "user-coordinates",
        }
    if address:
        return geocode_google(address, api_key=google_maps_api_key)
    return None

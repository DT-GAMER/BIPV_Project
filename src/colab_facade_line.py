"""Colab helpers for selecting a facade measurement line on a map."""

from __future__ import annotations

from .geospatial import _haversine_m


def facade_line_config_from_points(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> dict:
    """Return AnalysisConfig keyword values for a clicked facade line."""

    return {
        "facade_start_latitude": float(start_lat),
        "facade_start_longitude": float(start_lon),
        "facade_end_latitude": float(end_lat),
        "facade_end_longitude": float(end_lon),
    }


def create_facade_line_selector(
    center_lat: float,
    center_lon: float,
    *,
    zoom: int = 19,
):
    """Create an ipyleaflet map for drawing one facade-width line.

    Returns ``(map_widget, state, config_getter)``. Draw a line along the facade;
    ``state`` is updated with the selected endpoints and width in metres. Use
    ``config_getter()`` to pass the selected line into ``AnalysisConfig``.
    """

    try:
        from IPython.display import display
        from ipywidgets import Output
        from ipyleaflet import DrawControl, LayersControl, Map, Marker, basemaps
    except ImportError as exc:
        raise ImportError(
            "Interactive facade-line selection needs ipyleaflet and ipywidgets. "
            "Install with `pip install ipyleaflet ipywidgets` or paste the two "
            "facade endpoint coordinates manually."
        ) from exc

    state: dict = {
        "start": None,
        "end": None,
        "width_m": None,
    }
    output = Output()
    map_widget = Map(
        center=(center_lat, center_lon),
        zoom=zoom,
        basemap=basemaps.CartoDB.Positron,
        scroll_wheel_zoom=True,
    )
    map_widget.add_layer(Marker(location=(center_lat, center_lon)))
    map_widget.add_control(LayersControl())

    draw_control = DrawControl(
        polyline={
            "shapeOptions": {
                "color": "#e31a1c",
                "weight": 6,
            }
        },
        polygon={},
        rectangle={},
        circlemarker={},
        circle={},
    )
    map_widget.add_control(draw_control)

    def handle_draw(_target, action, geo_json):
        if action not in {"created", "edited"}:
            return
        geometry = geo_json.get("geometry") or {}
        if geometry.get("type") != "LineString":
            return
        coords = geometry.get("coordinates") or []
        if len(coords) < 2:
            return
        start_lon, start_lat = coords[0][:2]
        end_lon, end_lat = coords[-1][:2]
        width_m = _haversine_m(start_lat, start_lon, end_lat, end_lon)
        state.update(
            {
                "start": (float(start_lat), float(start_lon)),
                "end": (float(end_lat), float(end_lon)),
                "width_m": float(width_m),
            }
        )
        with output:
            output.clear_output()
            print(f"Selected facade width: {width_m:.2f} m")
            print(f"Start: {start_lat:.7f}, {start_lon:.7f}")
            print(f"End:   {end_lat:.7f}, {end_lon:.7f}")

    draw_control.on_draw(handle_draw)

    def get_config():
        if state["start"] is None or state["end"] is None:
            raise ValueError("Draw a facade line on the map before calling get_config().")
        start_lat, start_lon = state["start"]
        end_lat, end_lon = state["end"]
        return facade_line_config_from_points(start_lat, start_lon, end_lat, end_lon)

    display(map_widget, output)
    return map_widget, state, get_config

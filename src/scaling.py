"""Stage 9: real-world scaling from Google Earth or known dimensions."""

from __future__ import annotations

from .area import calculate_real_world_dimensions
from .geometry import validate_google_earth_dimensions


def estimate_real_world_scale(
    aligned_facade,
    window_boxes_np,
    facade_mask,
    ge_width_m: float | None = None,
    ge_height_m: float | None = None,
    require_google_earth_dimensions: bool = False,
    known_floors: int | None = None,
    floor_height_m: float = 3.0,
):
    """Validate and calculate real-world facade dimensions."""

    validation = validate_google_earth_dimensions(
        aligned_facade,
        window_boxes_np,
        ge_width_m,
        ge_height_m,
        floor_height_m,
        facade_mask=facade_mask,
        require_google_earth_dimensions=require_google_earth_dimensions,
    )
    dimensions = calculate_real_world_dimensions(
        aligned_facade,
        window_boxes_np,
        known_floors=known_floors,
        reference_height_m=floor_height_m,
        validated_width_m=validation["width_m"],
        validated_height_m=validation["height_m"],
        facade_mask=facade_mask,
    )
    return dimensions, validation

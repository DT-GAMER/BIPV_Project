"""Stage 9: real-world scaling from measured or inferred dimensions."""

from __future__ import annotations

from .area import calculate_real_world_dimensions
from .geometry import validate_google_earth_dimensions
from .scale_estimation import estimate_scale_from_image, validate_scale_estimate


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
    """Estimate real-world facade dimensions.

    If Google Earth dimensions are explicitly required, they become the scale
    source. Otherwise, the default mode is automatic image-based estimation, with
    optional Google Earth values used only for validation.
    """

    if not require_google_earth_dimensions:
        scale_estimate = estimate_scale_from_image(
            aligned_facade,
            facade_mask,
            window_boxes_np,
            known_floors=known_floors,
            default_floor_height_m=floor_height_m,
        )
        validation = validate_scale_estimate(scale_estimate, ge_width_m, ge_height_m)
        dimensions = {
            "num_floors": scale_estimate["num_floors"],
            "height_m": scale_estimate["height_m"],
            "width_m": scale_estimate["width_m"],
            "pixels_per_meter": scale_estimate["pixels_per_meter"],
            "total_facade_area_m2": scale_estimate["total_facade_area_m2"],
            "scale_source": scale_estimate["source"],
            "scale_confidence": scale_estimate["confidence"],
            "scale_method": scale_estimate["method"],
        }
        return dimensions, validation

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

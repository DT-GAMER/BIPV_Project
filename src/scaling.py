"""Stage 9: real-world scaling from measured or inferred dimensions."""

from __future__ import annotations

from .area import mask_extent
from .scale_estimation import estimate_scale_from_image, validate_scale_estimate


def estimate_real_world_scale(
    aligned_facade,
    window_boxes_np,
    facade_mask,
    window_mask=None,
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
            window_mask=window_mask,
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
            "floor_count_source": scale_estimate["floor_count_source"],
            "floor_count_candidates": scale_estimate["floor_count_candidates"],
        }
        return dimensions, validation

    if ge_width_m is None or ge_height_m is None:
        raise ValueError(
            "Google Earth dimensions are required. Set ge_width_m and ge_height_m "
            "before running calibrated area calculations."
        )

    image_scale_estimate = estimate_scale_from_image(
        aligned_facade,
        facade_mask,
        window_boxes_np,
        window_mask=window_mask,
        known_floors=known_floors,
        default_floor_height_m=floor_height_m,
    )
    validation = validate_scale_estimate(image_scale_estimate, ge_width_m, ge_height_m)
    validation.update(
        {
            "source": "google-earth",
            "calibration_source": "google-earth",
            "image_estimate_source": image_scale_estimate["source"],
            "image_estimate_height_m": image_scale_estimate["height_m"],
            "image_estimate_width_m": image_scale_estimate["width_m"],
            "image_estimate_area_m2": image_scale_estimate["total_facade_area_m2"],
            "reference_width_m": ge_width_m,
            "reference_height_m": ge_height_m,
            "floor_count_source": image_scale_estimate["floor_count_source"],
            "floor_count_candidates": image_scale_estimate["floor_count_candidates"],
        }
    )

    facade_height_px, _ = mask_extent(facade_mask)
    pixels_per_meter = facade_height_px / ge_height_m if ge_height_m else 0
    dimensions = {
        "num_floors": image_scale_estimate["num_floors"],
        "height_m": ge_height_m,
        "width_m": ge_width_m,
        "pixels_per_meter": pixels_per_meter,
        "total_facade_area_m2": ge_height_m * ge_width_m,
        "scale_source": "google-earth",
        "scale_confidence": validation.get("confidence"),
        "scale_method": "google-earth-calibrated-image-scale",
        "floor_count_source": image_scale_estimate["floor_count_source"],
        "floor_count_candidates": image_scale_estimate["floor_count_candidates"],
    }
    return dimensions, validation

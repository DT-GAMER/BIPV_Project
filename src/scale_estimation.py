"""Automatic real-world scale estimation for single facade images.

This module estimates facade dimensions without manual user input. It uses
detected facade geometry, floor/window structure, and typical architectural
priors. If measured Google Earth dimensions are later supplied, they are used
only as validation/calibration references.
"""

from __future__ import annotations

import numpy as np

from .area import count_floors_from_windows, mask_extent


def _count_floors(window_boxes_np, known_floors: int | None) -> int:
    if known_floors is not None:
        return int(known_floors)
    if len(window_boxes_np) > 0:
        return count_floors_from_windows(window_boxes_np)
    return 5


def _estimate_floor_height_m(window_boxes_np, default_floor_height_m: float) -> tuple[float, str]:
    """Estimate floor height prior from visible window proportions."""

    if len(window_boxes_np) < 3:
        return default_floor_height_m, "default-floor-height"

    median_window_h_norm = float(np.median(window_boxes_np[:, 3]))

    # Typical window height is around 1.2-1.6 m. The ratio between window height
    # and floor height often lands near 0.35-0.55 for residential/apartment blocks.
    if median_window_h_norm < 0.035:
        return max(default_floor_height_m, 3.2), "small-window-office-prior"
    if median_window_h_norm > 0.090:
        return min(default_floor_height_m, 2.8), "large-window-residential-prior"
    return default_floor_height_m, "default-floor-height"


def estimate_scale_from_image(
    aligned_facade,
    facade_mask,
    window_boxes_np,
    known_floors: int | None = None,
    default_floor_height_m: float = 3.0,
) -> dict:
    """Estimate facade width/height from image evidence and priors.

    The output is an estimate, not a physical measurement. It is intended for
    no-manual-input mode where the user only uploads an image.
    """

    facade_height_px, facade_width_px = mask_extent(facade_mask)
    num_floors = _count_floors(window_boxes_np, known_floors)
    floor_height_m, floor_height_source = _estimate_floor_height_m(
        window_boxes_np,
        default_floor_height_m,
    )

    height_m = num_floors * floor_height_m
    pixels_per_meter = facade_height_px / height_m if height_m else 0
    width_m = facade_width_px / pixels_per_meter if pixels_per_meter else 0

    confidence = 0.45
    evidence = []
    if known_floors is not None:
        confidence += 0.25
        evidence.append("manual-known-floor-count")
    elif len(window_boxes_np) >= 8:
        confidence += 0.20
        evidence.append("detected-window-floor-count")
    elif len(window_boxes_np) > 0:
        confidence += 0.10
        evidence.append("limited-window-floor-count")
    else:
        evidence.append("default-floor-count")

    if facade_mask.sum() > 0:
        confidence += 0.10
        evidence.append("facade-mask")

    confidence = min(confidence, 0.90)

    return {
        "source": "automatic-image-scale-estimate",
        "method": "floor-count-facade-aspect-prior",
        "confidence": confidence,
        "evidence": evidence,
        "num_floors": num_floors,
        "floor_height_m": floor_height_m,
        "floor_height_source": floor_height_source,
        "height_m": height_m,
        "width_m": width_m,
        "pixels_per_meter": pixels_per_meter,
        "total_facade_area_m2": height_m * width_m,
    }


def validate_scale_estimate(scale_estimate: dict, ge_width_m=None, ge_height_m=None) -> dict:
    """Compare automatic scale estimate against measured dimensions if available."""

    validation = {
        "source": scale_estimate["source"],
        "method": scale_estimate["method"],
        "confidence": scale_estimate["confidence"],
        "has_reference": ge_width_m is not None and ge_height_m is not None,
    }

    if ge_width_m is None or ge_height_m is None:
        validation["status"] = "estimated-no-reference"
        return validation

    height_error = abs(scale_estimate["height_m"] - ge_height_m) / ge_height_m * 100
    width_error = abs(scale_estimate["width_m"] - ge_width_m) / ge_width_m * 100
    area_ref = ge_width_m * ge_height_m
    area_error = (
        abs(scale_estimate["total_facade_area_m2"] - area_ref) / area_ref * 100
        if area_ref
        else 0
    )
    max_error = max(height_error, width_error, area_error)
    status = "excellent" if max_error < 5 else "acceptable" if max_error < 10 else "needs-calibration"

    validation.update(
        {
            "status": status,
            "reference_source": "google-earth-manual-validation",
            "reference_width_m": ge_width_m,
            "reference_height_m": ge_height_m,
            "height_error_percent": height_error,
            "width_error_percent": width_error,
            "area_error_percent": area_error,
        }
    )
    return validation

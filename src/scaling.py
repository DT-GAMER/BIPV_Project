"""Stage 9: real-world scaling from measured or inferred dimensions."""

from __future__ import annotations

import cv2
import numpy as np

from .area import mask_extent
from .scale_estimation import estimate_scale_from_image, validate_scale_estimate


def _house_facade_aspect_floor_prior(facade_mask) -> tuple[int | None, dict]:
    """Infer house floors from facade aspect ratio.

    Detached houses are usually wide relative to their height. When roof/gable
    artifacts produce many false window rows, the facade aspect ratio is a more
    stable low-rise cue than raw opening bands.
    """

    facade_height_px, facade_width_px = mask_extent(facade_mask)
    if facade_height_px <= 0 or facade_width_px <= 0:
        return None, {"reason": "empty-facade"}

    aspect = facade_height_px / max(facade_width_px, 1)
    if aspect <= 0.95:
        return 2, {"reason": "wide-low-rise-house", "facade_aspect_h_over_w": aspect}
    if aspect <= 1.35:
        return 3, {"reason": "moderately-tall-house", "facade_aspect_h_over_w": aspect}
    return None, {"reason": "aspect-not-low-rise", "facade_aspect_h_over_w": aspect}


def _house_opening_floor_prior(window_mask, facade_mask) -> tuple[int | None, dict]:
    """Infer house floors from filtered opening rows."""

    if window_mask is None or facade_mask.sum() == 0 or window_mask.sum() == 0:
        return None, {"reason": "missing-window-mask"}

    facade_height_px, facade_width_px = mask_extent(facade_mask)
    facade_area = int(facade_mask.sum())
    ys_facade, xs_facade = np.where(facade_mask)
    top = int(ys_facade.min())
    bottom = int(ys_facade.max())
    left = int(xs_facade.min())
    right = int(xs_facade.max())
    facade_h = max(bottom - top + 1, 1)
    facade_w = max(right - left + 1, 1)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        (window_mask & facade_mask).astype(np.uint8),
        connectivity=8,
    )

    centers = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < max(10, facade_area * 0.0004):
            continue
        if area > facade_area * 0.08:
            continue
        if w > facade_w * 0.38 or h > facade_h * 0.32:
            continue
        aspect = h / max(w, 1)
        if not 0.25 <= aspect <= 5.5:
            continue
        cx, cy = centroids[label_id]
        cy_norm = (cy - top) / facade_h
        # Ignore likely roof/gutter artifacts very close to the top.
        if cy_norm < 0.10:
            continue
        centers.append(float(cy_norm))

    if len(centers) < 2:
        return None, {"reason": "too-few-valid-openings", "valid_openings": len(centers)}

    centers = sorted(centers)
    bands = [[centers[0]]]
    for center in centers[1:]:
        if center - float(np.mean(bands[-1])) > 0.22:
            bands.append([])
        bands[-1].append(center)

    # Drop sparse single-artifact bands if stronger bands exist.
    counts = [len(band) for band in bands]
    max_count = max(counts)
    kept = [band for band, count in zip(bands, counts) if count >= max(2, max_count * 0.35)]
    floors = len(kept)
    if 2 <= floors <= 4:
        return floors, {
            "reason": "filtered-house-opening-bands",
            "valid_openings": len(centers),
            "raw_bands": len(bands),
            "kept_bands": floors,
            "band_counts": counts,
        }
    return None, {
        "reason": "opening-bands-out-of-range",
        "valid_openings": len(centers),
        "raw_bands": len(bands),
        "kept_bands": floors,
        "band_counts": counts,
    }


def _apply_house_floor_prior(
    scale_estimate: dict,
    facade_mask,
    window_mask=None,
    house_max_floors: int | None = None,
) -> dict:
    """Prefer automatic low-rise floor inference for detached houses."""

    candidates = scale_estimate.get("floor_count_candidates") or {}
    opening_floors, opening_quality = _house_opening_floor_prior(window_mask, facade_mask)
    aspect_floors, aspect_quality = _house_facade_aspect_floor_prior(facade_mask)

    if opening_floors is not None and opening_floors <= 3:
        house_floors = opening_floors
        source = "house-mode-opening-row-prior"
    elif aspect_floors is not None:
        house_floors = aspect_floors
        source = "house-mode-facade-aspect-prior"
    elif opening_floors is not None:
        house_floors = opening_floors
        source = "house-mode-opening-row-prior"
    elif house_max_floors is not None and scale_estimate.get("num_floors", 0) > house_max_floors:
        house_floors = int(house_max_floors)
        source = "house-mode-user-floor-cap"
    else:
        return scale_estimate

    if house_max_floors is not None:
        house_floors = min(house_floors, int(house_max_floors))

    if house_floors == scale_estimate.get("num_floors"):
        return scale_estimate

    facade_height_px, facade_width_px = mask_extent(facade_mask)
    floor_height_m = float(scale_estimate.get("floor_height_m", 3.1))
    height_m = house_floors * floor_height_m
    pixels_per_meter = facade_height_px / height_m if height_m else 0
    width_m = facade_width_px / pixels_per_meter if pixels_per_meter else 0

    updated = dict(scale_estimate)
    updated.update(
        {
            "num_floors": house_floors,
            "height_m": height_m,
            "width_m": width_m,
            "pixels_per_meter": pixels_per_meter,
            "total_facade_area_m2": height_m * width_m,
            "floor_count_source": source,
            "house_mode_floor_override": {
                "original_num_floors": scale_estimate.get("num_floors"),
                "selected_num_floors": house_floors,
                "house_max_floors": house_max_floors,
                "candidates": candidates,
                "opening_prior": opening_quality,
                "aspect_prior": aspect_quality,
            },
        }
    )
    updated["confidence"] = min(float(updated.get("confidence", 0.65)) + 0.05, 0.85)
    return updated


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
    building_type: str = "urban",
    house_max_floors: int | None = None,
):
    """Estimate real-world facade dimensions.

    If Google Earth dimensions are supplied, they become the metric scale source.
    Otherwise, the default mode is automatic image-based estimation. Setting
    require_google_earth_dimensions=True only controls whether missing reference
    dimensions should raise an error.
    """

    has_google_earth_reference = ge_width_m is not None and ge_height_m is not None

    if not require_google_earth_dimensions and not has_google_earth_reference:
        scale_estimate = estimate_scale_from_image(
            aligned_facade,
            facade_mask,
            window_boxes_np,
            window_mask=window_mask,
            known_floors=known_floors,
            default_floor_height_m=floor_height_m,
        )
        if known_floors is None and building_type.strip().lower() == "house":
            scale_estimate = _apply_house_floor_prior(
                scale_estimate,
                facade_mask,
                window_mask=window_mask,
                house_max_floors=house_max_floors,
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
            "floor_height_m": scale_estimate["floor_height_m"],
            "floor_height_source": scale_estimate["floor_height_source"],
            "house_mode_floor_override": scale_estimate.get("house_mode_floor_override"),
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
    if known_floors is None and building_type.strip().lower() == "house":
        image_scale_estimate = _apply_house_floor_prior(
            image_scale_estimate,
            facade_mask,
            window_mask=window_mask,
            house_max_floors=house_max_floors,
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
        "floor_height_m": image_scale_estimate["floor_height_m"],
        "floor_height_source": image_scale_estimate["floor_height_source"],
        "house_mode_floor_override": image_scale_estimate.get("house_mode_floor_override"),
    }
    return dimensions, validation

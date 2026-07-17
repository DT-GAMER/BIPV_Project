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


def _refine_geospatial_width_from_image_estimate(
    geospatial_reference: dict,
    image_width_m: float,
) -> tuple[dict, str]:
    """Prefer a footprint edge that agrees with the visible street facade.

    Overture/OSM may return a full building block where the longest edge is not
    the photographed frontage. When no camera/facade bearing is provided, combine
    the image-only width estimate with the distance from the input/geocoded point
    to each edge. The distance term biases the red verification edge toward the
    side of the building that is closest to the photographed/address side.
    """

    edges = geospatial_reference.get("footprint_edges") or []
    current_width = float(geospatial_reference.get("facade_width_m") or 0)
    if not edges or image_width_m <= 0 or current_width <= 0:
        return geospatial_reference, geospatial_reference.get(
            "facade_width_source", "geospatial-footprint-edge"
        )

    distances = [
        float(edge.get("distance_to_query_m"))
        for edge in edges
        if edge.get("distance_to_query_m") is not None
    ]
    max_distance = max(distances) if distances else 1.0

    def edge_score(edge):
        length = float(edge.get("length_m", 0))
        length_error = abs(length - image_width_m) / max(image_width_m, 1e-6)
        distance = edge.get("distance_to_query_m")
        distance_error = (
            float(distance) / max(max_distance, 1e-6)
            if distance is not None
            else 0.5
        )
        # Width still matters most, but the selected verification line should
        # favor the facade side closest to the address/photo point.
        return length_error + 0.35 * distance_error

    selected = min(edges, key=edge_score)
    selected_width = float(selected.get("length_m", 0))
    if selected_width <= 0:
        return geospatial_reference, geospatial_reference.get(
            "facade_width_source", "geospatial-footprint-edge"
        )

    current_error = abs(current_width - image_width_m) / max(image_width_m, 1e-6)
    selected_error = abs(selected_width - image_width_m) / max(image_width_m, 1e-6)
    current_score = edge_score(
        {
            "length_m": current_width,
            "distance_to_query_m": (
                geospatial_reference.get("facade_edge") or {}
            ).get("distance_to_query_m"),
        }
    )
    selected_score = edge_score(selected)
    if selected_score + 0.03 >= current_score and selected_error + 0.05 >= current_error:
        return geospatial_reference, geospatial_reference.get(
            "facade_width_source", "geospatial-footprint-edge"
        )

    refined = dict(geospatial_reference)
    refined["facade_width_m"] = selected_width
    refined["facade_edge"] = selected
    refined["facade_width_source"] = (
        f"{geospatial_reference.get('source', 'geospatial')}-image-width-matched-edge"
    )
    refined["width_refinement"] = {
        "original_width_m": current_width,
        "image_estimate_width_m": image_width_m,
        "selected_width_m": selected_width,
        "original_relative_error": current_error,
        "selected_relative_error": selected_error,
        "selection_score": selected_score,
        "original_selection_score": current_score,
        "selected_distance_to_query_m": selected.get("distance_to_query_m"),
    }
    return refined, refined["facade_width_source"]


def _height_from_geospatial_width_and_facade_aspect(
    facade_mask,
    geo_width_m: float,
) -> tuple[float, dict]:
    """Estimate facade height from measured footprint width and image aspect.

    Overture/OSM usually provides a reliable plan-view footprint but not a
    facade height. Once the photographed facade is aligned, the visible facade
    mask gives a pixel height-to-width ratio. Multiplying that ratio by the
    measured facade width produces a height estimate that does not depend on
    floor counting.
    """

    facade_height_px, facade_width_px = mask_extent(facade_mask)
    if facade_width_px <= 0 or geo_width_m <= 0:
        return 0.0, {
            "status": "failed",
            "reason": "missing-facade-width",
            "facade_height_px": int(facade_height_px),
            "facade_width_px": int(facade_width_px),
            "geo_width_m": float(geo_width_m or 0),
        }

    aspect_h_over_w = facade_height_px / max(facade_width_px, 1)
    return geo_width_m * aspect_h_over_w, {
        "status": "estimated",
        "method": "geospatial-width-facade-pixel-aspect",
        "facade_height_px": int(facade_height_px),
        "facade_width_px": int(facade_width_px),
        "facade_aspect_h_over_w": float(aspect_h_over_w),
        "reference_width_m": float(geo_width_m),
    }


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
    geospatial_reference: dict | None = None,
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
        if geospatial_reference is not None:
            geospatial_reference, geospatial_width_method = (
                _refine_geospatial_width_from_image_estimate(
                    geospatial_reference,
                    float(scale_estimate.get("width_m", 0)),
                )
            )
            facade_height_px, facade_width_px = mask_extent(facade_mask)
            geo_width_m = geospatial_reference.get("facade_width_m")
            source_geo_height_m = geospatial_reference.get("height_m")
            aspect_height_m, aspect_height_details = (
                _height_from_geospatial_width_and_facade_aspect(
                    facade_mask,
                    float(geo_width_m or 0),
                )
            )
            geo_height_m = source_geo_height_m or aspect_height_m or scale_estimate["height_m"]
            if geo_width_m and geo_height_m:
                pixels_per_meter_y = facade_height_px / geo_height_m if geo_height_m else 0
                pixels_per_meter_x = facade_width_px / geo_width_m if geo_width_m else 0
                if source_geo_height_m:
                    geo_confidence = 0.88
                    height_method = "geospatial-source-height"
                    height_source = geospatial_reference.get("height_source")
                elif aspect_height_m:
                    geo_confidence = 0.82
                    height_method = "geospatial-width-facade-pixel-aspect"
                    height_source = "facade-aspect-from-geospatial-width"
                else:
                    geo_confidence = 0.78
                    height_method = "floor-count-fallback"
                    height_source = scale_estimate["floor_height_source"]
                validation = validate_scale_estimate(scale_estimate, geo_width_m, geo_height_m)
                validation.update(
                    {
                        "source": "geospatial",
                        "calibration_source": geospatial_reference.get("source"),
                        "reference_width_m": geo_width_m,
                        "reference_height_m": geo_height_m,
                        "image_estimate_source": scale_estimate["source"],
                        "image_estimate_height_m": scale_estimate["height_m"],
                        "image_estimate_width_m": scale_estimate["width_m"],
                        "image_estimate_area_m2": scale_estimate["total_facade_area_m2"],
                        "height_estimate_source": height_source,
                        "height_estimate_method": height_method,
                        "aspect_height_estimate_m": aspect_height_m,
                        "floor_count_height_estimate_m": scale_estimate["height_m"],
                        "height_estimate_details": aspect_height_details,
                        "geospatial_reference": geospatial_reference,
                    }
                )
                dimensions = {
                    "num_floors": geospatial_reference.get("building_levels") or scale_estimate["num_floors"],
                    "height_m": geo_height_m,
                    "width_m": geo_width_m,
                    "pixels_per_meter": pixels_per_meter_y,
                    "pixels_per_meter_x": pixels_per_meter_x,
                    "pixels_per_meter_y": pixels_per_meter_y,
                    "total_facade_area_m2": geo_height_m * geo_width_m,
                    "scale_source": "geospatial",
                    "scale_confidence": geo_confidence,
                    "scale_method": (
                        "geospatial-footprint-calibrated-image-scale"
                        if source_geo_height_m
                        else "geospatial-width-aspect-calibrated-image-scale"
                    ),
                    "facade_width_source": geospatial_width_method,
                    "height_source": height_source,
                    "height_method": height_method,
                    "height_estimate_details": aspect_height_details,
                    "floor_count_height_estimate_m": scale_estimate["height_m"],
                    "floor_count_source": (
                        geospatial_reference.get("height_source")
                        or scale_estimate["floor_count_source"]
                    ),
                    "floor_count_candidates": scale_estimate["floor_count_candidates"],
                    "floor_height_m": scale_estimate["floor_height_m"],
                    "floor_height_source": scale_estimate["floor_height_source"],
                    "house_mode_floor_override": scale_estimate.get("house_mode_floor_override"),
                    "geospatial_reference": geospatial_reference,
                }
                return dimensions, validation
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

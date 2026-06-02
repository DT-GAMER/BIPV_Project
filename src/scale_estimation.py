"""Automatic real-world scale estimation for single facade images.

This module estimates facade dimensions without manual user input. It uses
detected facade geometry, floor/window structure, and typical architectural
priors. If measured Google Earth dimensions are later supplied, they are used
only as validation/calibration references.
"""

from __future__ import annotations

import numpy as np
import cv2

from .area import count_floors_from_windows, mask_extent


def _count_floor_bands_from_centers(centers_y, min_floor_gap: float) -> int:
    centers_y = np.sort(np.asarray(centers_y, dtype=float))
    if len(centers_y) < 2:
        return len(centers_y) if len(centers_y) > 0 else 0
    bands = [[centers_y[0]]]
    for center_y in centers_y[1:]:
        if center_y - np.mean(bands[-1]) > min_floor_gap:
            bands.append([])
        bands[-1].append(center_y)
    return len(bands)


def _floor_band_centers(centers_y, min_floor_gap: float) -> list[float]:
    centers_y = np.sort(np.asarray(centers_y, dtype=float))
    if len(centers_y) == 0:
        return []
    bands = [[centers_y[0]]]
    for center_y in centers_y[1:]:
        if center_y - np.mean(bands[-1]) > min_floor_gap:
            bands.append([])
        bands[-1].append(center_y)
    return [float(np.mean(band)) for band in bands]


def _robust_floor_bands(mask_boxes: np.ndarray, adaptive_gap: float) -> list[float]:
    """Return floor band centers after discarding sparse bands.

    Bands with fewer than 40% of the median per-band window count are treated
    as artifacts (cornices, transoms, parapets) rather than real floors.
    """
    band_centers = _floor_band_centers(mask_boxes[:, 1], adaptive_gap)
    if len(band_centers) <= 1:
        return band_centers

    window_ys = mask_boxes[:, 1]
    half = adaptive_gap * 0.65
    counts = [max(1, int(np.sum(np.abs(window_ys - c) < half))) for c in band_centers]
    median_count = float(np.median(counts))
    threshold = median_count * 0.40

    return [c for c, n in zip(band_centers, counts) if n >= threshold]


def _facade_y_extent_norm(facade_mask) -> tuple[float, float] | None:
    if facade_mask is None or facade_mask.sum() == 0:
        return None
    height = facade_mask.shape[0]
    ys = np.where(facade_mask)[0]
    return float(ys.min() / height), float(ys.max() / height)


def _extrapolate_floors_from_facade_height(
    band_centers: list[float],
    facade_mask,
) -> int:
    """Infer missing top/ground bands from facade height and row spacing."""

    if len(band_centers) < 2:
        return len(band_centers)

    facade_extent = _facade_y_extent_norm(facade_mask)
    if facade_extent is None:
        return len(band_centers)

    y_min, y_max = facade_extent
    row_gap = float(np.median(np.diff(np.sort(band_centers))))
    if row_gap <= 0:
        return len(band_centers)

    top_margin = max(0.0, band_centers[0] - y_min)
    bottom_margin = max(0.0, y_max - band_centers[-1])
    extrapolated = len(band_centers)
    if top_margin > 0.75 * row_gap:
        extrapolated += 1
    if bottom_margin > 0.75 * row_gap:
        extrapolated += 1
    return int(np.clip(extrapolated, len(band_centers), 20))


def _window_boxes_from_mask(window_mask, facade_mask) -> np.ndarray:
    """Approximate window boxes from the final segmented window mask."""

    if window_mask is None or window_mask.sum() == 0:
        return np.empty((0, 4), dtype=float)

    height, width = window_mask.shape
    facade_area = int(facade_mask.sum()) if facade_mask is not None else height * width
    min_area = max(20, int(facade_area * 0.00008))
    max_area = max(min_area + 1, int(facade_area * 0.040))

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        window_mask.astype(np.uint8),
        connectivity=8,
    )

    boxes = []
    for label in range(1, num_labels):
        x, y, component_width, component_height, area = stats[label]
        if not (min_area <= area <= max_area):
            continue
        if component_width < 3 or component_height < 6:
            continue
        if component_width / max(component_height, 1) > 8:
            continue
        cx, cy = centroids[label]
        boxes.append(
            [
                float(cx / width),
                float(cy / height),
                float(component_width / width),
                float(component_height / height),
            ]
        )
    return np.array(boxes, dtype=float)


def _count_floors(
    window_boxes_np,
    known_floors: int | None,
    window_mask=None,
    facade_mask=None,
) -> tuple[int, str, dict]:
    if known_floors is not None:
        return int(known_floors), "manual-known-floor-count", {"manual": int(known_floors)}

    candidates = {}
    if len(window_boxes_np) > 0:
        candidates["dino_window_boxes"] = count_floors_from_windows(window_boxes_np)

    mask_boxes = _window_boxes_from_mask(window_mask, facade_mask)
    if len(mask_boxes) > 0:
        median_h = float(np.median(mask_boxes[:, 3]))
        adaptive_gap = float(np.clip(median_h * 1.35, 0.035, 0.085))
        # Use robust band detection: drops sparse bands caused by cornices,
        # transoms, or parapets that are mistaken for window rows.
        mask_floor_bands = _robust_floor_bands(mask_boxes, adaptive_gap)
        candidates["segmented_window_mask"] = len(mask_floor_bands)
        # Cap extrapolation at +1 over the direct (filtered) band count.
        candidates["segmented_window_mask_with_facade_extent"] = min(
            _extrapolate_floors_from_facade_height(mask_floor_bands, facade_mask),
            len(mask_floor_bands) + 1,
        )

    if not candidates:
        return 5, "default-floor-count", {"default": 5}

    values = sorted(candidates.values())
    n = len(values)
    if n % 2 == 1:
        median_floors = values[n // 2]
    else:
        median_floors = int(round((values[n // 2 - 1] + values[n // 2]) / 2.0))
    median_floors = int(np.clip(median_floors, 2, 20))

    source = min(candidates, key=lambda k: abs(candidates[k] - median_floors))
    return median_floors, source, candidates


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
    window_mask=None,
    known_floors: int | None = None,
    default_floor_height_m: float = 3.0,
) -> dict:
    """Estimate facade width/height from image evidence and priors.

    The output is an estimate, not a physical measurement. It is intended for
    no-manual-input mode where the user only uploads an image.
    """

    facade_height_px, facade_width_px = mask_extent(facade_mask)
    num_floors, floor_count_source, floor_count_candidates = _count_floors(
        window_boxes_np,
        known_floors,
        window_mask=window_mask,
        facade_mask=facade_mask,
    )
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
    elif floor_count_source.startswith("segmented_window_mask"):
        confidence += 0.25
        evidence.append("segmented-window-mask-floor-count")
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
        "floor_count_source": floor_count_source,
        "floor_count_candidates": floor_count_candidates,
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

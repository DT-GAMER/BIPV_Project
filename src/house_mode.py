"""Optional post-processing for detached/low-rise house facades."""

from __future__ import annotations

import cv2
import numpy as np


def _top_facade_zone(facade_mask: np.ndarray, fraction: float = 0.36) -> np.ndarray:
    zone = np.zeros_like(facade_mask, dtype=bool)
    if facade_mask.sum() == 0:
        return zone

    ys, _ = np.where(facade_mask)
    top = int(ys.min())
    bottom = int(ys.max())
    facade_height = max(bottom - top + 1, 1)
    limit = min(facade_mask.shape[0], top + int(round(facade_height * fraction)))
    zone[top:limit, :] = True
    return zone


def _detect_pitched_roof_pixels(aligned_facade: np.ndarray, facade_mask: np.ndarray) -> np.ndarray:
    """Detect dark roof-like pixels in the top facade band.

    The result is used as an exclusion mask, not to reshape the facade wall.
    This keeps house-mode behavior safer than directly cutting the facade mask.
    """

    if facade_mask.sum() == 0:
        return np.zeros_like(facade_mask, dtype=bool)

    hsv = cv2.cvtColor(aligned_facade, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(aligned_facade, cv2.COLOR_RGB2GRAY)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    top_zone = _top_facade_zone(facade_mask)
    roof_like = (
        ((value < 150) & (saturation < 95))
        | ((hue >= 88) & (hue <= 135) & (saturation > 30) & (value < 190))
        | (gray < 105)
    )
    candidate = roof_like & top_zone & facade_mask

    facade_area = int(facade_mask.sum())
    cleaned = np.zeros_like(candidate, dtype=bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8),
        connectivity=8,
    )
    ys, _ = np.where(facade_mask)
    facade_top = int(ys.min())
    facade_height = int(ys.max() - ys.min() + 1)

    for label_id in range(1, num_labels):
        x, y, component_w, component_h, area = stats[label_id]
        if area < max(25, facade_area * 0.002):
            continue
        if area > facade_area * 0.16:
            continue
        if component_w < 10:
            continue
        if component_h > facade_height * 0.28:
            continue
        if y > facade_top + facade_height * 0.30:
            continue
        cleaned |= labels == label_id

    # If the detector wants to remove too much, it is probably seeing dark stone
    # or shadow rather than roof. Keep the original segmentation in that case.
    if cleaned.sum() > facade_area * 0.18:
        return np.zeros_like(facade_mask, dtype=bool)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    cleaned_uint = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
    return cleaned_uint.astype(bool) & facade_mask


def _regularize_small_openings(opening_mask: np.ndarray, facade_mask: np.ndarray) -> np.ndarray:
    """Optional small-house opening cleanup.

    This is intentionally disabled by default because irregular/arched windows
    can be valid architecture. When enabled, it only runs on simple facades with
    a small number of detected openings.
    """

    if opening_mask.sum() == 0 or facade_mask.sum() == 0:
        return opening_mask.astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opening_mask.astype(np.uint8),
        connectivity=8,
    )
    component_count = num_labels - 1
    if component_count == 0 or component_count > 14:
        return opening_mask.astype(bool)

    facade_area = int(facade_mask.sum())
    regularized = np.zeros_like(opening_mask, dtype=bool)
    changed = False

    for label_id in range(1, num_labels):
        x, y, component_w, component_h, area = stats[label_id]
        bbox_area = component_w * component_h
        if bbox_area <= 0:
            continue

        area_fraction = bbox_area / max(facade_area, 1)
        aspect = component_h / max(component_w, 1)
        fill = area / bbox_area

        if 0.0005 <= area_fraction <= 0.08 and 0.25 <= aspect <= 4.5 and fill >= 0.22:
            candidate = np.zeros_like(opening_mask, dtype=bool)
            candidate[y : y + component_h, x : x + component_w] = True
            if (candidate & facade_mask).sum() / max(candidate.sum(), 1) >= 0.65:
                regularized |= candidate & facade_mask
                changed = True
                continue

        regularized |= labels == label_id

    if not changed:
        return opening_mask.astype(bool)

    ratio = regularized.sum() / max(opening_mask.sum(), 1)
    if not 0.70 <= ratio <= 1.65:
        return opening_mask.astype(bool)
    return regularized & facade_mask


def apply_house_mode_postprocessing(
    segmentation: dict,
    aligned_facade: np.ndarray,
    regularize_openings: bool = False,
) -> tuple[dict, dict]:
    """Apply optional house-specific semantic cleanup.

    House mode does not alter the main facade mask. It adds roof-like top areas
    to the roof exclusion mask and optionally regularizes simple openings.
    """

    facade_mask = segmentation["facade_mask"].astype(bool)
    roof_mask = segmentation.get("roof_mask", np.zeros_like(facade_mask)).astype(bool)
    house_roof_mask = _detect_pitched_roof_pixels(aligned_facade, facade_mask)

    updated = dict(segmentation)
    updated["roof_mask"] = (roof_mask | house_roof_mask) & facade_mask

    if regularize_openings:
        updated["window_mask"] = _regularize_small_openings(
            updated["window_mask"] & facade_mask,
            facade_mask,
        )
        if "raw_window_mask" in updated:
            updated["raw_window_mask"] = _regularize_small_openings(
                updated["raw_window_mask"] & facade_mask,
                facade_mask,
            )

    quality = dict(updated.get("quality", {}))
    quality["house_mode"] = {
        "enabled": True,
        "roof_exclusion_pixels_added": int(house_roof_mask.sum()),
        "regularize_openings": bool(regularize_openings),
        "mode": "roof-exclusion-only" if not regularize_openings else "roof-and-opening-cleanup",
    }
    quality["roof_exclusion_pixels"] = int(updated["roof_mask"].sum())
    updated["quality"] = quality

    return updated, quality["house_mode"]

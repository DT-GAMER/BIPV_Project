"""Facade, window, door, and balcony segmentation."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision.ops import nms as torchvision_nms

from .detection import preprocess_for_dino
from .utils import decode_box


def project_bbox_through_warp(xmin, ymin, xmax, ymax, matrix, height: int, width: int):
    corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    x_vals = np.clip(dst[:, 0], 0, width - 1)
    y_vals = np.clip(dst[:, 1], 0, height - 1)
    return int(x_vals.min()), int(y_vals.min()), int(x_vals.max()), int(y_vals.max())


def boxes_to_xyxy(boxes_normalized, height: int, width: int):
    cx = boxes_normalized[:, 0] * width
    cy = boxes_normalized[:, 1] * height
    box_width = boxes_normalized[:, 2] * width
    box_height = boxes_normalized[:, 3] * height
    return torch.stack([cx - box_width / 2, cy - box_height / 2, cx + box_width / 2, cy + box_height / 2], dim=1)


def apply_nms_per_class(boxes_normalized, logits, phrases, height: int, width: int, iou: float = 0.35):
    if len(boxes_normalized) == 0:
        return []

    boxes_px = boxes_to_xyxy(boxes_normalized, height, width)
    keep = []
    for cls in set(phrase.lower().strip() for phrase in phrases):
        indices = [index for index, phrase in enumerate(phrases) if cls in phrase.lower()]
        if indices:
            kept = torchvision_nms(boxes_px[indices].float(), logits[indices].float(), iou)
            keep.extend(indices[index] for index in kept.tolist())
    return sorted(set(keep))


def detect_facade_elements(aligned_facade, dino_model, device: str):
    """Detect windows, doors, entrances, and balconies on the rectified facade."""

    from groundingdino.util.inference import predict as dino_predict

    image_tensor = preprocess_for_dino(aligned_facade, device)
    boxes_raw, logits_raw, phrases_raw = None, None, None
    caption = (
        "window . glass window . facade window . door . entrance . balcony . "
        "wall . roof edge"
    )
    for box_threshold, text_threshold in [(0.22, 0.16), (0.18, 0.12), (0.14, 0.10), (0.10, 0.08)]:
        boxes, logits, phrases = dino_predict(
            model=dino_model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if boxes_raw is None:
            boxes_raw, logits_raw, phrases_raw = boxes, logits, phrases
        if sum(1 for phrase in phrases if "window" in phrase.lower()) >= 3:
            boxes_raw, logits_raw, phrases_raw = boxes, logits, phrases
            break
    return boxes_raw, logits_raw, phrases_raw


def _add_sam_window_fallback(
    auto_masks,
    facade_mask,
    existing_window_mask,
    door_mask,
    balcony_mask,
    min_window_detections: int,
    current_window_count: int,
):
    """Add conservative SAM-derived window candidates when DINO misses windows."""

    if current_window_count >= min_window_detections:
        return existing_window_mask, 0

    height, width = facade_mask.shape
    facade_area = facade_mask.sum()
    if facade_area == 0:
        return existing_window_mask, 0

    fallback_mask = existing_window_mask.copy()
    added = 0
    occupied = door_mask | balcony_mask

    for mask_data in auto_masks:
        segment = mask_data["segmentation"]
        if segment.shape != (height, width):
            continue

        area = segment.sum()
        if not (0.0015 * facade_area <= area <= 0.060 * facade_area):
            continue

        inside = (segment & facade_mask).sum()
        if inside / max(area, 1) < 0.75:
            continue

        if (segment & occupied).sum() / max(area, 1) > 0.25:
            continue

        ys, xs = np.where(segment)
        if len(xs) == 0:
            continue

        bbox_w = xs.max() - xs.min() + 1
        bbox_h = ys.max() - ys.min() + 1
        aspect = bbox_h / max(bbox_w, 1)
        fill_ratio = area / max(bbox_w * bbox_h, 1)

        # Windows can vary, but extreme slivers and very sparse blobs are risky.
        if not (0.45 <= aspect <= 3.50):
            continue
        if fill_ratio < 0.25:
            continue

        fallback_mask |= segment
        added += 1

    return fallback_mask, added


def _add_cv_window_fallback(
    aligned_facade,
    facade_mask,
    existing_window_mask,
    door_mask,
    balcony_mask,
    min_area_fraction: float = 0.00020,
    max_area_fraction: float = 0.02000,
):
    """Detect glass-like rectangular window candidates missed by DINO/SAM."""

    height, width = facade_mask.shape
    facade_area = facade_mask.sum()
    if facade_area == 0:
        return existing_window_mask, 0

    hsv = cv2.cvtColor(aligned_facade, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(aligned_facade, cv2.COLOR_RGB2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    glass_like = ((saturation < 85) & (value > 80) & (value < 245)) | (gray < 75)
    glass_like &= facade_mask
    glass_like &= ~(door_mask | balcony_mask | existing_window_mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    cleaned = cv2.morphologyEx(glass_like.astype(np.uint8) * 255, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fallback_mask = existing_window_mask.copy()
    added = 0

    for contour in contours:
        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
        bbox_area = bbox_w * bbox_h
        if bbox_area == 0:
            continue

        area_fraction = bbox_area / max(facade_area, 1)
        if not (min_area_fraction <= area_fraction <= max_area_fraction):
            continue

        aspect = bbox_h / max(bbox_w, 1)
        if not (0.60 <= aspect <= 7.0):
            continue

        contour_area = cv2.contourArea(contour)
        if contour_area / bbox_area < 0.35:
            continue

        candidate = np.zeros((height, width), dtype=bool)
        candidate[y : y + bbox_h, x : x + bbox_w] = True
        if (candidate & facade_mask).sum() / max(candidate.sum(), 1) < 0.90:
            continue

        fallback_mask |= candidate
        added += 1

    return fallback_mask, added


def _component_boxes_from_mask(mask, facade_mask):
    height, width = mask.shape
    facade_area = int(facade_mask.sum())
    if facade_area == 0 or mask.sum() == 0:
        return []

    min_area = max(20, int(facade_area * 0.00008))
    max_area = max(min_area + 1, int(facade_area * 0.035))
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    boxes = []
    for label in range(1, num_labels):
        x, y, box_w, box_h, area = stats[label]
        if not (min_area <= area <= max_area):
            continue
        if box_w < 4 or box_h < 6:
            continue
        aspect = box_h / max(box_w, 1)
        if not (0.35 <= aspect <= 5.5):
            continue
        cx, cy = centroids[label]
        boxes.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(box_w),
                "h": int(box_h),
                "cx": float(cx),
                "cy": float(cy),
            }
        )
    return boxes


def _cluster_positions(values, tolerance):
    values = sorted(float(value) for value in values)
    if not values:
        return []

    clusters = [[values[0]]]
    for value in values[1:]:
        if abs(value - np.mean(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.mean(cluster)) for cluster in clusters]


def _facade_extent(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _extend_regular_centers(centers, lower_bound, upper_bound, max_extra: int = 2):
    """Extend repeated grid centers into likely hidden top/bottom rows."""

    centers = sorted(float(center) for center in centers)
    if len(centers) < 2:
        return centers

    gap = float(np.median(np.diff(centers)))
    if gap <= 1:
        return centers

    extended = list(centers)
    for _ in range(max_extra):
        candidate = extended[0] - gap
        if candidate < lower_bound:
            break
        if extended[0] - lower_bound > 0.45 * gap:
            extended.insert(0, candidate)
        else:
            break

    for _ in range(max_extra):
        candidate = extended[-1] + gap
        if candidate > upper_bound:
            break
        if upper_bound - extended[-1] > 0.45 * gap:
            extended.append(candidate)
        else:
            break

    return extended


def _add_grid_inferred_windows(
    existing_window_mask,
    facade_mask,
    door_mask,
    balcony_mask,
    reconstructed_mask=None,
):
    """Infer windows hidden by removed obstacles from regular facade structure."""

    if reconstructed_mask is None or reconstructed_mask.sum() == 0:
        return existing_window_mask, 0

    boxes = _component_boxes_from_mask(existing_window_mask, facade_mask)
    if len(boxes) < 6:
        return existing_window_mask, 0

    median_w = float(np.median([box["w"] for box in boxes]))
    median_h = float(np.median([box["h"] for box in boxes]))
    if median_w <= 0 or median_h <= 0:
        return existing_window_mask, 0

    x_centers = _cluster_positions(
        [box["cx"] for box in boxes],
        tolerance=max(8.0, median_w * 0.70),
    )
    y_centers = _cluster_positions(
        [box["cy"] for box in boxes],
        tolerance=max(8.0, median_h * 0.85),
    )
    if len(x_centers) < 2 or len(y_centers) < 2:
        return existing_window_mask, 0

    height, width = existing_window_mask.shape
    extent = _facade_extent(facade_mask)
    if extent is not None:
        fx1, fy1, fx2, fy2 = extent
        x_centers = _extend_regular_centers(
            x_centers,
            fx1 + median_w / 2,
            fx2 - median_w / 2,
            max_extra=1,
        )
        y_centers = _extend_regular_centers(
            y_centers,
            fy1 + median_h / 2,
            fy2 - median_h / 2,
            max_extra=2,
        )

    inferred = existing_window_mask.copy()
    occupied = existing_window_mask | door_mask | balcony_mask
    added = 0

    candidate_w = int(max(4, round(median_w)))
    candidate_h = int(max(6, round(median_h)))

    for cy in y_centers:
        for cx in x_centers:
            x1 = int(round(cx - candidate_w / 2))
            y1 = int(round(cy - candidate_h / 2))
            x2 = int(round(cx + candidate_w / 2))
            y2 = int(round(cy + candidate_h / 2))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            candidate = np.zeros((height, width), dtype=bool)
            candidate[y1:y2, x1:x2] = True
            candidate_area = int(candidate.sum())
            if candidate_area == 0:
                continue

            if (candidate & facade_mask).sum() / candidate_area < 0.70:
                continue
            if (candidate & occupied).sum() / candidate_area > 0.20:
                continue
            if (candidate & reconstructed_mask).sum() / candidate_area < 0.08:
                continue

            inferred |= candidate
            occupied |= candidate
            added += 1

    return inferred, added


def _nearest_center(value, centers):
    if not centers:
        return None
    return min(centers, key=lambda center: abs(center - value))


def _draw_window_rect(mask, facade_mask, cx, cy, box_w, box_h):
    height, width = mask.shape
    x1 = int(round(cx - box_w / 2))
    y1 = int(round(cy - box_h / 2))
    x2 = int(round(cx + box_w / 2))
    y2 = int(round(cy + box_h / 2))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return False

    candidate = np.zeros_like(mask, dtype=bool)
    candidate[y1:y2, x1:x2] = True
    candidate_area = int(candidate.sum())
    if candidate_area == 0:
        return False
    if (candidate & facade_mask).sum() / candidate_area < 0.65:
        return False

    mask |= candidate & facade_mask
    return True


def _regularize_window_grid(
    existing_window_mask,
    facade_mask,
    door_mask,
    balcony_mask,
    reconstructed_mask=None,
):
    """Convert detected windows into a cleaner row/column engineering grid.

    The visual inpainting output may be imperfect, especially behind trees and
    cars. For area estimation, repeated facade structure is more reliable: use
    visible windows to infer rows/columns, snap detections to that grid, then add
    missing window openings only where a removed obstacle likely hid them.
    """

    boxes = _component_boxes_from_mask(existing_window_mask, facade_mask)
    if len(boxes) < 4:
        return existing_window_mask, {
            "regularized": False,
            "regularized_windows": 0,
            "regularized_inferred_windows": 0,
            "grid_rows": 0,
            "grid_columns": 0,
            "reason": "not-enough-window-components",
        }

    widths = np.array([box["w"] for box in boxes], dtype=float)
    heights = np.array([box["h"] for box in boxes], dtype=float)
    median_w = float(np.median(widths))
    median_h = float(np.median(heights))
    if median_w <= 0 or median_h <= 0:
        return existing_window_mask, {
            "regularized": False,
            "regularized_windows": 0,
            "regularized_inferred_windows": 0,
            "grid_rows": 0,
            "grid_columns": 0,
            "reason": "invalid-window-size",
        }

    # Build a grid from robust center clusters. The tolerances are intentionally
    # forgiving because rectification is approximate for street-level photos.
    x_centers = _cluster_positions(
        [box["cx"] for box in boxes],
        tolerance=max(10.0, median_w * 0.90),
    )
    y_centers = _cluster_positions(
        [box["cy"] for box in boxes],
        tolerance=max(10.0, median_h * 0.95),
    )

    extent = _facade_extent(facade_mask)
    if extent is not None:
        fx1, fy1, fx2, fy2 = extent
        x_centers = _extend_regular_centers(
            x_centers,
            fx1 + median_w / 2,
            fx2 - median_w / 2,
            max_extra=2,
        )
        y_centers = _extend_regular_centers(
            y_centers,
            fy1 + median_h / 2,
            fy2 - median_h / 2,
            max_extra=2,
        )

    if len(x_centers) < 2 or len(y_centers) < 2:
        return existing_window_mask, {
            "regularized": False,
            "regularized_windows": 0,
            "regularized_inferred_windows": 0,
            "grid_rows": len(y_centers),
            "grid_columns": len(x_centers),
            "reason": "weak-grid",
        }

    height, width = existing_window_mask.shape
    regularized = np.zeros((height, width), dtype=bool)
    occupied = door_mask | balcony_mask
    grid_hits = {}
    row_support = {index: 0 for index in range(len(y_centers))}
    col_support = {index: 0 for index in range(len(x_centers))}
    drawn_windows = 0

    rect_w = int(np.clip(round(median_w), 5, max(width, 5)))
    rect_h = int(np.clip(round(median_h), 7, max(height, 7)))
    max_snap_dx = max(12.0, median_w * 1.25)
    max_snap_dy = max(12.0, median_h * 1.25)

    for box in boxes:
        snapped_x = _nearest_center(box["cx"], x_centers)
        snapped_y = _nearest_center(box["cy"], y_centers)
        if snapped_x is None or snapped_y is None:
            continue
        if abs(snapped_x - box["cx"]) > max_snap_dx:
            continue
        if abs(snapped_y - box["cy"]) > max_snap_dy:
            continue

        col = int(np.argmin([abs(center - snapped_x) for center in x_centers]))
        row = int(np.argmin([abs(center - snapped_y) for center in y_centers]))
        grid_hits[(row, col)] = True
        row_support[row] += 1
        col_support[col] += 1

        local_w = int(np.clip(round(0.5 * box["w"] + 0.5 * median_w), 5, median_w * 1.45))
        local_h = int(np.clip(round(0.5 * box["h"] + 0.5 * median_h), 7, median_h * 1.45))
        if _draw_window_rect(regularized, facade_mask, snapped_x, snapped_y, local_w, local_h):
            drawn_windows += 1

    if drawn_windows < max(4, int(len(boxes) * 0.65)):
        return existing_window_mask, {
            "regularized": False,
            "regularized_windows": drawn_windows,
            "regularized_inferred_windows": 0,
            "grid_rows": len(y_centers),
            "grid_columns": len(x_centers),
            "median_window_width_px": median_w,
            "median_window_height_px": median_h,
            "reason": "grid-snap-mismatch",
        }

    inferred_added = 0
    if reconstructed_mask is not None and reconstructed_mask.sum() > 0:
        min_row_support = max(2, int(len(x_centers) * 0.20))
        min_col_support = max(2, int(len(y_centers) * 0.20))
        for row, cy in enumerate(y_centers):
            for col, cx in enumerate(x_centers):
                if (row, col) in grid_hits:
                    continue
                if row_support.get(row, 0) < min_row_support:
                    continue
                if col_support.get(col, 0) < min_col_support:
                    continue

                candidate = np.zeros((height, width), dtype=bool)
                x1 = int(round(cx - rect_w / 2))
                y1 = int(round(cy - rect_h / 2))
                x2 = int(round(cx + rect_w / 2))
                y2 = int(round(cy + rect_h / 2))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                candidate[y1:y2, x1:x2] = True
                candidate_area = int(candidate.sum())
                if candidate_area == 0:
                    continue

                if (candidate & facade_mask).sum() / candidate_area < 0.65:
                    continue
                if (candidate & occupied).sum() / candidate_area > 0.15:
                    continue
                # Only create new windows in areas that were actually hidden by
                # an obstacle/removal mask. This avoids inventing full grids on
                # plain walls with intentionally irregular fenestration.
                if (candidate & reconstructed_mask).sum() / candidate_area < 0.10:
                    continue

                regularized |= candidate & facade_mask
                inferred_added += 1

    # Keep the final engineering mask grid-shaped. If the grid is too weak, the
    # function returns the original mask earlier instead of forcing rectangles.
    regularized = cv2.morphologyEx(
        regularized.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    regularized &= facade_mask & ~occupied

    if regularized.sum() < existing_window_mask.sum() * 0.35:
        return existing_window_mask, {
            "regularized": False,
            "regularized_windows": drawn_windows,
            "regularized_inferred_windows": 0,
            "grid_rows": len(y_centers),
            "grid_columns": len(x_centers),
            "median_window_width_px": median_w,
            "median_window_height_px": median_h,
            "reason": "regularized-mask-too-small",
        }
    if regularized.sum() > existing_window_mask.sum() * 2.60:
        return existing_window_mask, {
            "regularized": False,
            "regularized_windows": drawn_windows,
            "regularized_inferred_windows": 0,
            "grid_rows": len(y_centers),
            "grid_columns": len(x_centers),
            "median_window_width_px": median_w,
            "median_window_height_px": median_h,
            "reason": "regularized-mask-too-large",
        }

    return regularized, {
        "regularized": True,
        "regularized_windows": drawn_windows,
        "regularized_inferred_windows": inferred_added,
        "grid_rows": len(y_centers),
        "grid_columns": len(x_centers),
        "median_window_width_px": median_w,
        "median_window_height_px": median_h,
        "reason": "ok",
    }


def _is_plausible_facade_candidate(candidate, building_bbox_mask):
    bbox_area = int(building_bbox_mask.sum())
    candidate_area = int(candidate.sum())
    if bbox_area == 0 or candidate_area == 0:
        return False

    coverage = candidate_area / bbox_area
    if not (0.35 <= coverage <= 1.10):
        return False

    candidate_extent = _facade_extent(candidate)
    bbox_extent = _facade_extent(building_bbox_mask)
    if candidate_extent is None or bbox_extent is None:
        return False

    cx1, cy1, cx2, cy2 = candidate_extent
    bx1, by1, bx2, by2 = bbox_extent
    candidate_w = max(cx2 - cx1 + 1, 1)
    candidate_h = max(cy2 - cy1 + 1, 1)
    bbox_w = max(bx2 - bx1 + 1, 1)
    bbox_h = max(by2 - by1 + 1, 1)

    if candidate_w / bbox_w < 0.70 or candidate_h / bbox_h < 0.60:
        return False

    # Reject very sparse/fragmented facade masks. For engineering area, a
    # stable footprint is preferable to an irregular hallucinated SAM region.
    rectangularity = candidate_area / max(candidate_w * candidate_h, 1)
    return rectangularity >= 0.45


def segment_facade_components(
    aligned_facade,
    mask_generator,
    predictor,
    dino_model,
    device: str,
    building_bbox,
    transform_matrix,
    min_window_detections: int = 3,
    use_cv_window_fallback: bool = True,
    cv_window_min_area_fraction: float = 0.00020,
    cv_window_max_area_fraction: float = 0.02000,
    reconstructed_mask=None,
):
    """Create facade, window, door, and balcony masks."""

    height, width = aligned_facade.shape[:2]
    if transform_matrix is not None:
        bx1, by1, bx2, by2 = project_bbox_through_warp(*building_bbox, transform_matrix, height, width)
    else:
        bx1, by1, bx2, by2 = 0, 0, width - 1, height - 1

    slack_x, slack_y = int(width * 0.05), int(height * 0.05)
    bx1, by1 = max(0, bx1 - slack_x), max(0, by1 - slack_y)
    bx2, by2 = min(width - 1, bx2 + slack_x), min(height - 1, by2 + slack_y)

    building_bbox_mask = np.zeros((height, width), dtype=bool)
    building_bbox_mask[by1:by2, bx1:bx2] = True

    auto_masks = mask_generator.generate(aligned_facade)
    facade_mask = None
    facade_source = "building_bbox"
    for mask in sorted(auto_masks, key=lambda item: item["area"], reverse=True)[:8]:
        candidate = mask["segmentation"] & building_bbox_mask
        if _is_plausible_facade_candidate(candidate, building_bbox_mask):
            facade_mask = candidate
            facade_source = "sam_plausible_candidate"
            break
    if facade_mask is None:
        facade_mask = building_bbox_mask.copy()

    boxes_raw, logits_raw, phrases_raw = detect_facade_elements(aligned_facade, dino_model, device)
    kept = apply_nms_per_class(boxes_raw, logits_raw, phrases_raw, height, width)
    boxes = boxes_raw[kept]
    logits = logits_raw[kept]
    phrases = [phrases_raw[index] for index in kept]

    filtered_boxes, filtered_logits, filtered_phrases = [], [], []
    for box, logit, phrase in zip(boxes, logits, phrases):
        cx_px = int(np.clip(box[0].item() * width, 0, width - 1))
        cy_px = int(np.clip(box[1].item() * height, 0, height - 1))
        if facade_mask[cy_px, cx_px] or building_bbox_mask[cy_px, cx_px]:
            filtered_boxes.append(box)
            filtered_logits.append(logit)
            filtered_phrases.append(phrase)

    boxes = torch.stack(filtered_boxes) if filtered_boxes else boxes_raw[:0]
    logits = torch.stack(filtered_logits) if filtered_logits else logits_raw[:0]
    phrases = filtered_phrases

    clean_boxes, clean_logits, clean_phrases = [], [], []
    for box, logit, phrase in zip(boxes, logits, phrases):
        cx, cy, box_width, box_height = box.tolist()
        phrase_lower = phrase.lower()
        if "balcony" in phrase_lower and (box_width * box_height > 0.20 or box_height > 0.60):
            continue
        if "window" in phrase_lower and (box_width > 0.20 or box_width * box_height > 0.08):
            continue
        clean_boxes.append(box)
        clean_logits.append(logit)
        clean_phrases.append(phrase)

    boxes = torch.stack(clean_boxes) if clean_boxes else boxes_raw[:0]
    logits = torch.stack(clean_logits) if clean_logits else logits_raw[:0]
    phrases = clean_phrases

    predictor.set_image(aligned_facade)
    window_mask = np.zeros((height, width), dtype=bool)
    door_mask = np.zeros((height, width), dtype=bool)
    balcony_mask = np.zeros((height, width), dtype=bool)

    for box, phrase in zip(boxes, phrases):
        input_box = decode_box(box.cpu().numpy(), height, width)
        phrase_lower = phrase.lower()
        if "balcony" in phrase_lower:
            masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
            box_area_px = (input_box[2] - input_box[0]) * (input_box[3] - input_box[1])
            valid = [(mask, score) for mask, score in zip(masks, scores) if 100 < mask.sum() < 0.50 * box_area_px]
            best = min(valid, key=lambda item: item[0].sum())[0] if valid else masks[np.argmin([mask.sum() for mask in masks])]
            balcony_mask |= best
        elif "window" in phrase_lower:
            mask, _, _ = predictor.predict(box=input_box, multimask_output=False)
            window_mask |= mask[0]
        elif "door" in phrase_lower or "entrance" in phrase_lower:
            mask, _, _ = predictor.predict(box=input_box, multimask_output=False)
            door_mask |= mask[0]

    window_count = sum(1 for phrase in phrases if "window" in phrase.lower())
    window_mask, fallback_windows_added = _add_sam_window_fallback(
        auto_masks,
        facade_mask,
        window_mask,
        door_mask,
        balcony_mask,
        min_window_detections,
        window_count,
    )
    cv_windows_added = 0
    if use_cv_window_fallback:
        window_mask, cv_windows_added = _add_cv_window_fallback(
            aligned_facade,
            facade_mask,
            window_mask,
            door_mask,
            balcony_mask,
            min_area_fraction=cv_window_min_area_fraction,
            max_area_fraction=cv_window_max_area_fraction,
        )
    raw_window_mask = window_mask.copy()
    window_mask, grid_quality = _regularize_window_grid(
        window_mask,
        facade_mask,
        door_mask,
        balcony_mask,
        reconstructed_mask=reconstructed_mask,
    )
    grid_inferred_windows_added = grid_quality["regularized_inferred_windows"]

    return {
        "facade_mask": facade_mask,
        "window_mask": window_mask,
        "raw_window_mask": raw_window_mask,
        "door_mask": door_mask,
        "balcony_mask": balcony_mask,
        "boxes": boxes,
        "logits": logits,
        "phrases": phrases,
        "building_bbox_mask": building_bbox_mask,
        "auto_masks": auto_masks,
        "quality": {
            "dino_window_count": window_count,
            "sam_fallback_windows_added": fallback_windows_added,
            "cv_fallback_windows_added": cv_windows_added,
            "grid_inferred_windows_added": grid_inferred_windows_added,
            "grid_regularized": grid_quality["regularized"],
            "grid_regularization_reason": grid_quality["reason"],
            "grid_regularized_windows": grid_quality["regularized_windows"],
            "grid_regularized_inferred_windows": grid_quality["regularized_inferred_windows"],
            "grid_rows": grid_quality["grid_rows"],
            "grid_columns": grid_quality["grid_columns"],
            "median_window_width_px": grid_quality.get("median_window_width_px"),
            "median_window_height_px": grid_quality.get("median_window_height_px"),
            "facade_coverage_percent": 100 * facade_mask.sum() / max(height * width, 1),
            "facade_mask_source": facade_source,
        },
    }

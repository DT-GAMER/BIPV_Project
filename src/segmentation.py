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


def segment_facade_components(
    aligned_facade,
    mask_generator,
    predictor,
    dino_model,
    device: str,
    building_bbox,
    transform_matrix,
    min_window_detections: int = 3,
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
    for mask in sorted(auto_masks, key=lambda item: item["area"], reverse=True)[:5]:
        candidate = mask["segmentation"] & building_bbox_mask
        if candidate.sum() / max(building_bbox_mask.sum(), 1) > 0.10:
            facade_mask = candidate
            break
    if facade_mask is None:
        facade_mask = building_bbox_mask

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

    return {
        "facade_mask": facade_mask,
        "window_mask": window_mask,
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
            "facade_coverage_percent": 100 * facade_mask.sum() / max(height * width, 1),
        },
    }

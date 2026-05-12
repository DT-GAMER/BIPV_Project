"""Obstacle mask construction and inpainting."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .utils import combine_masks, decode_box


def build_robust_mask(
    raw_mask: np.ndarray,
    dilate_kernel: int = 9,
    dilate_iters: int = 1,
    shadow_pad_frac: float = 0.05,
    max_mask_fraction: float = 0.22,
    sparse_fill_threshold: float = 0.18,
    max_hull_bbox_fraction: float = 0.12,
) -> np.ndarray:
    """Expand object masks without letting branchy obstacles erase the facade.

    Tree branches often create sparse masks with a very large convex hull. Using
    that hull directly can turn a thin tree into a building-sized inpaint mask.
    This function only applies hull filling to compact components and falls
    back to a smaller raw-mask dilation if the final mask becomes too large.
    """

    uint_mask = (raw_mask * 255).astype(np.uint8)
    height, width = uint_mask.shape
    image_area = height * width

    n_labels, labels = cv2.connectedComponents(uint_mask)
    hull_mask = np.zeros_like(uint_mask)
    for label in range(1, n_labels):
        component = (labels == label).astype(np.uint8) * 255
        ys, xs = np.where(component > 0)
        if len(xs) == 0:
            continue

        bbox_w = xs.max() - xs.min() + 1
        bbox_h = ys.max() - ys.min() + 1
        bbox_area = bbox_w * bbox_h
        fill_ratio = len(xs) / max(bbox_area, 1)
        bbox_fraction = bbox_area / max(image_area, 1)

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if fill_ratio >= sparse_fill_threshold and bbox_fraction <= max_hull_bbox_fraction:
                hull = cv2.convexHull(contour)
                cv2.fillPoly(hull_mask, [hull], 255)
            else:
                cv2.drawContours(hull_mask, [contour], -1, 255, thickness=cv2.FILLED)

    n_labels, labels = cv2.connectedComponents(hull_mask)
    shadow_mask = hull_mask.copy()
    for label in range(1, n_labels):
        ys, xs = np.where(labels == label)
        if len(ys) == 0:
            continue
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        component_h = y_max - y_min + 1
        component_w = x_max - x_min + 1
        component_fraction = (component_h * component_w) / max(image_area, 1)
        if component_fraction <= max_hull_bbox_fraction:
            shadow_bottom = min(height - 1, y_max + int(component_h * shadow_pad_frac))
            shadow_mask[y_max:shadow_bottom, x_min:x_max] = 255

    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    expanded = cv2.dilate(shadow_mask, kernel, iterations=dilate_iters)

    if expanded.sum() / max(image_area * 255, 1) > max_mask_fraction:
        fallback_kernel = np.ones((max(3, dilate_kernel // 2), max(3, dilate_kernel // 2)), np.uint8)
        expanded = cv2.dilate(uint_mask, fallback_kernel, iterations=1)

    return expanded.astype(bool)


def segment_obstacles_with_sam(image_rgb, boxes, remove_ids, predictor) -> np.ndarray:
    """Use SAM box prompts to produce a combined obstacle mask."""

    height, width = image_rgb.shape[:2]
    predictor.set_image(image_rgb)

    raw_masks = []
    for index in remove_ids:
        input_box = decode_box(boxes[index].cpu().numpy(), height, width)
        mask, _, _ = predictor.predict(box=input_box, multimask_output=False)
        raw_masks.append(mask[0])

    return combine_masks(raw_masks, height, width)


def remove_obstacles(
    image_rgb,
    robust_mask,
    lama,
    sd_pipe=None,
    run_stable_diffusion: bool = True,
):
    """Remove obstacles with TELEA, LaMa, and optional Stable Diffusion refinement."""

    mask_uint8 = (robust_mask * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    telea_bgr = cv2.inpaint(image_bgr, mask_uint8, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    telea_rgb = cv2.cvtColor(telea_bgr, cv2.COLOR_BGR2RGB)

    lama_result = lama(Image.fromarray(telea_rgb), Image.fromarray(mask_uint8))
    lama_image = np.array(lama_result)

    if not run_stable_diffusion or sd_pipe is None:
        return lama_image

    sd_result = sd_pipe(
        prompt=(
            "clean building facade, matching brick wall, stone cladding, "
            "empty pavement, no vehicles, no people, no trees, photorealistic"
        ),
        negative_prompt=(
            "car, vehicle, tree, truck, bus, person, shadow, blur, watermark, "
            "unrealistic, flower, distorted"
        ),
        image=Image.fromarray(lama_image),
        mask_image=Image.fromarray(mask_uint8),
        strength=0.50,
        num_inference_steps=40,
        guidance_scale=8.5,
    ).images[0]
    return np.array(sd_result)

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
    close_kernel_scale: float = 2.0,
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
    close_size = max(3, int(round(dilate_kernel * close_kernel_scale)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    closed = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, close_kernel)
    expanded = cv2.dilate(closed, kernel, iterations=dilate_iters)

    if expanded.sum() / max(image_area * 255, 1) > max_mask_fraction:
        fallback_kernel_size = max(3, int(round(dilate_kernel * 0.75)))
        fallback_kernel = np.ones((fallback_kernel_size, fallback_kernel_size), np.uint8)
        expanded = cv2.dilate(closed, fallback_kernel, iterations=1)

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


def build_obstacle_box_mask(
    image_shape,
    boxes,
    remove_ids,
    phrases,
    pad_frac: float = 0.035,
    max_box_fraction: float = 0.08,
    max_width_fraction: float = 0.45,
    max_height_fraction: float = 0.35,
) -> np.ndarray:
    """Build compact DINO box supplements for obstacle masks.

    SAM can under-segment compact objects such as cars and poles. Full boxes are
    risky for trees, hedges, fences, and large foreground regions because they
    can erase the facade. Those large/irregular objects stay SAM-mask based and
    are expanded by ``build_robust_mask`` instead.
    """

    height, width = image_shape[:2]
    image_area = height * width
    box_mask = np.zeros((height, width), dtype=bool)
    for index in remove_ids:
        raw_x1, raw_y1, raw_x2, raw_y2 = decode_box(boxes[index].cpu().numpy(), height, width)
        phrase = phrases[index].lower()

        box_w = max(0.0, raw_x2 - raw_x1)
        box_h = max(0.0, raw_y2 - raw_y1)
        box_fraction = (box_w * box_h) / max(image_area, 1)
        width_fraction = box_w / max(width, 1)
        height_fraction = box_h / max(height, 1)

        compact_obstacle = any(
            keyword in phrase
            for keyword in ("car", "vehicle", "automobile", "truck", "bus", "person")
        )
        if not compact_obstacle:
            continue
        if box_fraction > max_box_fraction:
            continue
        if width_fraction > max_width_fraction or height_fraction > max_height_fraction:
            continue

        local_pad = pad_frac
        if any(keyword in phrase for keyword in ("car", "vehicle", "automobile", "truck", "bus")):
            local_pad = max(local_pad, 0.040)

        pad_x = int(width * local_pad)
        pad_y = int(height * local_pad)
        x1 = int(max(0, np.floor(raw_x1 - pad_x)))
        y1 = int(max(0, np.floor(raw_y1 - pad_y)))
        x2 = int(min(width - 1, np.ceil(raw_x2 + pad_x)))
        y2 = int(min(height - 1, np.ceil(raw_y2 + pad_y)))
        box_mask[y1 : y2 + 1, x1 : x2 + 1] = True

    return box_mask


def _match_reconstruction_to_context(image_rgb, mask_uint8, ring_kernel_size: int = 35):
    """Color-match and feather inpainted regions to nearby unmasked context."""

    mask_height, mask_width = mask_uint8.shape[:2]
    if image_rgb.shape[:2] != (mask_height, mask_width):
        image_rgb = cv2.resize(
            image_rgb,
            (mask_width, mask_height),
            interpolation=cv2.INTER_LINEAR,
        )

    mask = mask_uint8 > 0
    if mask.sum() == 0:
        return image_rgb

    refined = image_rgb.astype(np.float32).copy()
    original = refined.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ring_kernel_size, ring_kernel_size),
    )
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    context_ring = dilated & ~mask

    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    for label in range(1, n_labels):
        component = labels == label
        ys, xs = np.where(component)
        if len(xs) == 0:
            continue

        x1 = max(0, int(xs.min()) - ring_kernel_size)
        x2 = min(mask.shape[1], int(xs.max()) + ring_kernel_size + 1)
        y1 = max(0, int(ys.min()) - ring_kernel_size)
        y2 = min(mask.shape[0], int(ys.max()) + ring_kernel_size + 1)
        local_context = context_ring[y1:y2, x1:x2]
        local_component = component[y1:y2, x1:x2]
        if local_context.sum() < 25:
            continue

        patch = refined[y1:y2, x1:x2]
        context_pixels = original[y1:y2, x1:x2][local_context]
        inpaint_pixels = patch[local_component]
        if len(context_pixels) == 0 or len(inpaint_pixels) == 0:
            continue

        context_mean = context_pixels.mean(axis=0)
        context_std = context_pixels.std(axis=0) + 1e-6
        inpaint_mean = inpaint_pixels.mean(axis=0)
        inpaint_std = inpaint_pixels.std(axis=0) + 1e-6
        matched = (inpaint_pixels - inpaint_mean) * (context_std / inpaint_std) + context_mean
        patch[local_component] = np.clip(matched, 0, 255)
        refined[y1:y2, x1:x2] = patch

    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=3.0)
    alpha = np.clip(alpha[..., None], 0.0, 1.0)
    blended = refined * alpha + original * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def remove_obstacles(
    image_rgb,
    robust_mask,
    lama,
    sd_pipe=None,
    run_stable_diffusion: bool = True,
    removal_dilate_kernel: int = 3,
):
    """Remove obstacles with TELEA, LaMa, and optional Stable Diffusion refinement."""

    if removal_dilate_kernel > 1:
        kernel = np.ones((removal_dilate_kernel, removal_dilate_kernel), np.uint8)
        robust_mask = cv2.dilate(robust_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    mask_uint8 = (robust_mask * 255).astype(np.uint8)
    mask_uint8 = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    telea_bgr = cv2.inpaint(image_bgr, mask_uint8, inpaintRadius=11, flags=cv2.INPAINT_TELEA)
    telea_rgb = cv2.cvtColor(telea_bgr, cv2.COLOR_BGR2RGB)

    lama_result = lama(Image.fromarray(telea_rgb), Image.fromarray(mask_uint8))
    lama_image = np.array(lama_result)
    lama_image = _match_reconstruction_to_context(lama_image, mask_uint8)

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
    return _match_reconstruction_to_context(np.array(sd_result), mask_uint8)

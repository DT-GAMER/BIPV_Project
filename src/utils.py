"""Small shared utilities for masks and model boxes."""

from __future__ import annotations

import cv2
import numpy as np


def load_image(path):
    """Backward-compatible BGR image loader."""

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def decode_box(box_cxcywh: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert normalized cx/cy/w/h box coordinates to pixel xyxy."""

    cx, cy, box_width, box_height = box_cxcywh
    return np.array(
        [
            (cx - box_width / 2) * width,
            (cy - box_height / 2) * height,
            (cx + box_width / 2) * width,
            (cy + box_height / 2) * height,
        ]
    )


def combine_masks(masks: list[np.ndarray], height: int, width: int) -> np.ndarray:
    """Combine boolean masks with OR."""

    combined = np.zeros((height, width), dtype=bool)
    for mask in masks:
        combined |= mask
    return combined


def dilate_mask(
    mask: np.ndarray,
    kernel_size: int = 9,
    iterations: int = 1,
) -> np.ndarray:
    """Dilate a boolean mask and return a boolean mask."""

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)

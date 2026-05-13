"""Stage 7: final usable BIPV surface segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from .utils import dilate_mask


def warp_mask(mask, transform_matrix, output_shape):
    """Warp a binary mask into the aligned facade coordinate frame."""

    if transform_matrix is None:
        return mask.astype(bool)

    height, width = output_shape[:2]
    return cv2.warpPerspective(
        mask.astype("uint8"),
        transform_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def segment_bipv_surface(
    facade_mask,
    window_mask,
    door_mask,
    balcony_mask,
    shadow_mask=None,
    obstacle_mask=None,
    obstacle_exclusion_dilate_kernel: int = 15,
):
    """Build the final binary mask of usable BIPV installation surface."""

    usable_mask = facade_mask.copy()
    usable_mask &= ~dilate_mask(window_mask, kernel_size=5, iterations=1)
    usable_mask &= ~dilate_mask(door_mask, kernel_size=5, iterations=1)
    usable_mask &= ~dilate_mask(balcony_mask, kernel_size=7, iterations=1)

    if obstacle_mask is None:
        obstacle_exclusion_mask = np.zeros_like(facade_mask, dtype=bool)
    else:
        obstacle_exclusion_mask = dilate_mask(
            obstacle_mask,
            kernel_size=obstacle_exclusion_dilate_kernel,
            iterations=1,
        ) & facade_mask
        usable_mask &= ~obstacle_exclusion_mask

    if shadow_mask is None:
        shadow_mask = np.zeros_like(facade_mask, dtype=bool)

    usable_shadow_reduced = usable_mask & ~(shadow_mask & facade_mask)

    return {
        "usable_mask": usable_mask,
        "usable_mask_reduced": usable_shadow_reduced,
        "obstacle_exclusion_mask": obstacle_exclusion_mask,
    }

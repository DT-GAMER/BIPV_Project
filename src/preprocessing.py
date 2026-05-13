"""Stage 1: image acquisition and preprocessing."""

from __future__ import annotations

import cv2

from .image_io import load_image_rgb, resize_to_max_side


def normalize_image(image_rgb, max_side: int | None = None, denoise: bool = False):
    """Normalize a facade image for downstream CV stages."""

    image = resize_to_max_side(image_rgb, max_side)
    if denoise:
        image = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    return image


def load_and_preprocess_image(
    image_path: str,
    max_side: int | None = None,
    denoise: bool = False,
):
    """Load an RGB image and apply lightweight normalization."""

    return normalize_image(load_image_rgb(image_path), max_side=max_side, denoise=denoise)

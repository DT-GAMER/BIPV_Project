"""Image loading helpers."""

from __future__ import annotations

import cv2


def load_image_bgr(path: str):
    """Load an image in OpenCV BGR format."""

    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at '{path}'")
    return image


def load_image_rgb(path: str):
    """Load an image as RGB for model and matplotlib usage."""

    return cv2.cvtColor(load_image_bgr(path), cv2.COLOR_BGR2RGB)


def save_image_rgb(path: str, image_rgb) -> None:
    """Save an RGB image using OpenCV."""

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, image_bgr)
    if not ok:
        raise OSError(f"Could not save image to '{path}'")

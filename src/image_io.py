"""Image loading helpers."""

from __future__ import annotations

import cv2


def resize_to_max_side(image, max_side: int | None):
    """Resize image so its largest side is at most max_side."""

    if max_side is None:
        return image

    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return image

    scale = max_side / largest_side
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def load_image_bgr(path: str):
    """Load an image in OpenCV BGR format."""

    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at '{path}'")
    return image


def load_image_rgb(path: str, max_side: int | None = None):
    """Load an image as RGB for model and matplotlib usage."""

    image_rgb = cv2.cvtColor(load_image_bgr(path), cv2.COLOR_BGR2RGB)
    return resize_to_max_side(image_rgb, max_side)


def save_image_rgb(path: str, image_rgb) -> None:
    """Save an RGB image using OpenCV."""

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, image_bgr)
    if not ok:
        raise OSError(f"Could not save image to '{path}'")

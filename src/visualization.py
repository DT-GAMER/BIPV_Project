"""Matplotlib visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt


def show_image(image_rgb, title: str = "", figsize=(10, 6)) -> None:
    plt.figure(figsize=figsize)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_mask_overlay(
    image_rgb,
    mask,
    color=(255, 0, 0),
    title: str = "Mask Overlay",
) -> None:
    overlay = image_rgb.copy()
    overlay[mask] = color
    show_image(overlay, title)


def show_side_by_side(
    left,
    right,
    left_title: str = "Before",
    right_title: str = "After",
    figsize=(16, 8),
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for axis, image, title in zip(axes, [left, right], [left_title, right_title]):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    plt.show()

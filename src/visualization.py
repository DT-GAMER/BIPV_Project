"""Matplotlib visualization helpers."""

from __future__ import annotations

import numpy as np
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


def make_mask_overlay(image_rgb, mask, color=(0, 255, 0), alpha: float = 0.55):
    """Return an RGB image with a semi-transparent mask overlay."""

    overlay = image_rgb.copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = (
        (1 - alpha) * overlay[mask].astype(np.float32) + alpha * color_arr
    ).astype(np.uint8)
    return overlay


def make_binary_mask_image(mask, foreground=(230, 240, 255), background=(0, 0, 0)):
    """Convert a boolean mask to a displayable RGB image."""

    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[:, :] = background
    image[mask] = foreground
    return image


def workflow_images_from_result(result):
    """Create ordered stage images for one pipeline result."""

    return [
        ("Facade Image", result["image_rgb"]),
        (
            "Obstacle Detection",
            make_mask_overlay(result["image_rgb"], result["obstacle_mask"], color=(255, 0, 0)),
        ),
        ("Obstacle Removal", result["clean_image"]),
        ("Facade Alignment", result["aligned_facade"]),
        (
            "Segmentation Result",
            make_binary_mask_image(result["usable_results"]["usable_mask"]),
        ),
    ]


def show_workflow_grid(results, column_titles=None, figsize_per_cell=(3.0, 2.2)):
    """Display results like a stage-by-stage workflow figure.

    Rows are stages; columns are analysed images.
    """

    if isinstance(results, dict):
        results = [results]

    columns = len(results)
    stage_sets = [workflow_images_from_result(result) for result in results]
    rows = len(stage_sets[0])

    fig_width = max(5, columns * figsize_per_cell[0])
    fig_height = max(6, rows * figsize_per_cell[1])
    fig, axes = plt.subplots(rows, columns, figsize=(fig_width, fig_height), squeeze=False)

    for col, stage_images in enumerate(stage_sets):
        for row, (stage_name, image) in enumerate(stage_images):
            axis = axes[row][col]
            axis.imshow(image)
            axis.axis("off")
            if col == 0:
                axis.set_ylabel(stage_name, rotation=0, ha="right", va="center", labelpad=55)
            if row == 0:
                title = column_titles[col] if column_titles else f"Image {col + 1}"
                axis.set_title(title)

    plt.tight_layout()
    plt.show()


def show_bipv_scenario_bars(result, metric: str = "annual_kwh", title: str | None = None):
    """Plot paper-style BIPV scenario comparison bars."""

    scenario_data = result["bipv_scenarios"]["scenarios"]
    labels = ["None", "Shadow", "Window", "Both"]
    keys = ["none", "shadow", "window", "both"]
    values = [scenario_data[key][metric] for key in keys]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values, color=["#888888", "#8E5EA2", "#3C8DAD", "#2E7D32"])
    plt.bar_label(bars, fmt="%.1f", padding=3)
    plt.ylabel(metric.replace("_", " "))
    plt.title(title or f"BIPV Scenario Comparison ({metric})")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

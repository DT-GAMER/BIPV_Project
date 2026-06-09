"""Matplotlib visualization helpers."""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


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


def show_segmentation_alignment(result, figsize=(10, 7)) -> None:
    """Display facade/window/usable masks over the aligned facade for checking."""

    show_image(make_segmentation_alignment_image(result), "Segmentation Alignment Check", figsize=figsize)


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


def make_segmentation_alignment_image(result):
    """Return facade/window/usable masks overlaid on the aligned facade."""

    aligned = result["aligned_facade"]
    segmentation = result["segmentation"]
    usable_mask = result["usable_results"]["usable_mask"]

    overlay = aligned.copy()
    facade_mask = segmentation["facade_mask"]
    window_mask = segmentation["window_mask"]
    door_mask = segmentation["door_mask"]
    balcony_mask = segmentation["balcony_mask"]
    roof_mask = segmentation.get("roof_mask", np.zeros_like(facade_mask, dtype=bool))

    overlay[facade_mask] = (
        0.78 * overlay[facade_mask].astype(np.float32)
        + 0.22 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)
    overlay[usable_mask] = (
        0.68 * overlay[usable_mask].astype(np.float32)
        + 0.32 * np.array([0, 190, 0], dtype=np.float32)
    ).astype(np.uint8)
    overlay[window_mask] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[door_mask | balcony_mask | roof_mask] = np.array([255, 160, 0], dtype=np.uint8)
    return overlay


def _fit_image_to_canvas(image_rgb, canvas_size=(260, 190), background=(255, 255, 255)):
    """Resize an image into a fixed canvas without distorting aspect ratio."""

    target_w, target_h = canvas_size
    image = np.asarray(image_rgb)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return np.full((target_h, target_w, 3), background, dtype=np.uint8)

    scale = min(target_w / width, target_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), background, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _draw_method_cell(axis, row, stage_name):
    """Draw the compact method/sidebar cell used by the paper-style figure."""

    method_by_row = {
        0: "Input",
        1: "Grounding\nDINO",
        2: "SAM +\nInpainting",
        3: "Perspective\nTransform",
        4: "Grounding\nDINO + SAM",
    }
    icon_by_row = {
        0: "Facade\nImage",
        1: "Obstacle\nDetection",
        2: "Obstacle\nRemoval",
        3: "Facade\nAlignment",
        4: "Segmentation\nResult",
    }

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.42,
        0.67,
        method_by_row.get(row, stage_name),
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
        color="#16415f",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#d8eef7",
            "edgecolor": "#8abed4",
            "linewidth": 0.8,
        },
    )
    axis.text(
        0.42,
        0.30,
        icon_by_row.get(row, stage_name),
        ha="center",
        va="center",
        fontsize=7,
        color="#1d1d1d",
    )
    if row < 4:
        axis.annotate(
            "",
            xy=(0.86, -0.10),
            xytext=(0.86, 0.18),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#2f7da4", "linewidth": 1.5},
        )


def _bbox_from_mask(mask, padding_frac: float = 0.06):
    """Return [x1, y1, x2, y2] pixel bbox of the non-zero mask area with padding."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    pad_x = max(10, int((x2 - x1) * padding_frac))
    pad_y = max(10, int((y2 - y1) * padding_frac))
    return [max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w, x2 + pad_x), min(h, y2 + pad_y)]


def _crop_to_bbox(image_rgb, bbox):
    """Crop an image array to a [x1, y1, x2, y2] bounding box."""
    if bbox is None:
        return image_rgb
    img = np.asarray(image_rgb)
    h, w = img.shape[:2]
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return image_rgb
    return img[y1:y2, x1:x2]


def workflow_images_from_result(result, segmentation_view: str = "overlay"):
    """Create ordered stage images for one pipeline result.

    Each row is cropped tightly to the building — source rows (1-3) use the
    DINO-detected building bbox; rectified rows (4-5) use the facade mask bbox.
    This matches the reference paper layout where the building fills the cell.
    """

    # Source-image-space crop: use the perspective-transform source quad corners.
    # These are the exact 4 corners of the facade region used for the homography —
    # tighter and more accurate than the DINO bbox which includes adjacent buildings.
    source_bbox = None
    src_corners = result.get("src_corners")
    h_src, w_src = result["image_rgb"].shape[:2]
    if src_corners is not None and len(src_corners) >= 3:
        try:
            corners = np.asarray(src_corners)
            xs, ys = corners[:, 0], corners[:, 1]
            pad_x = max(10, int((xs.max() - xs.min()) * 0.04))
            pad_y = max(10, int((ys.max() - ys.min()) * 0.04))
            source_bbox = [
                max(0, int(xs.min()) - pad_x),
                max(0, int(ys.min()) - pad_y),
                min(w_src, int(xs.max()) + pad_x),
                min(h_src, int(ys.max()) + pad_y),
            ]
        except (TypeError, IndexError):
            pass
    if source_bbox is None:
        # Fallback to DINO building bbox if src_corners unavailable
        try:
            bb = result["stages"]["source_detection"]["source_building_bbox"]
            pad_x = max(10, int((bb[2] - bb[0]) * 0.04))
            pad_y = max(10, int((bb[3] - bb[1]) * 0.04))
            source_bbox = [
                max(0, int(bb[0]) - pad_x),
                max(0, int(bb[1]) - pad_y),
                min(w_src, int(bb[2]) + pad_x),
                min(h_src, int(bb[3]) + pad_y),
            ]
        except (KeyError, TypeError, IndexError):
            pass

    # Aligned-image-space crop: facade mask bounding box
    aligned_bbox = _bbox_from_mask(result["segmentation"]["facade_mask"], padding_frac=0.06)

    if segmentation_view == "overlay":
        segmentation_title = "Segmentation Overlay"
        segmentation_image = make_segmentation_alignment_image(result)
    else:
        segmentation_title = "Segmentation Result"
        segmentation_image = make_binary_mask_image(result["usable_results"]["usable_mask"])

    return [
        ("Facade Image",
         _crop_to_bbox(result["image_rgb"], source_bbox)),
        ("Obstacle Detection",
         _crop_to_bbox(
             make_mask_overlay(result["image_rgb"], result["obstacle_mask"], color=(255, 0, 0)),
             source_bbox,
         )),
        ("Obstacle Removal",
         _crop_to_bbox(result["clean_image"], source_bbox)),
        ("Facade Alignment",
         _crop_to_bbox(result["aligned_facade"], aligned_bbox)),
        (segmentation_title,
         _crop_to_bbox(segmentation_image, aligned_bbox)),
    ]


def build_workflow_grid_figure(
    results,
    column_titles=None,
    figsize_per_cell=(3.0, 2.2),
    label_each_panel: bool = True,
    paper_style: bool = True,
    show_method_column: bool = True,
    title: str | None = "Workflow and result for building facade RGB images parsing.",
    segmentation_view: str = "overlay",
):
    """Build a stage-by-stage workflow figure and return it."""
    if isinstance(results, dict):
        results = [results]

    columns = len(results)
    stage_sets = [
        workflow_images_from_result(result, segmentation_view=segmentation_view)
        for result in results
    ]
    rows = len(stage_sets[0])

    extra_columns = 1 if paper_style and show_method_column else 0
    fig_width = max(5, (columns + extra_columns * 0.85) * figsize_per_cell[0])
    fig_height = max(6, rows * figsize_per_cell[1])
    if title:
        fig_height += 0.45

    if paper_style and show_method_column:
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
        grid = gridspec.GridSpec(
            rows,
            columns + 1,
            figure=fig,
            width_ratios=[0.72] + [1.0] * columns,
            wspace=0.08,
            hspace=0.10,
        )
        axes = np.empty((rows, columns), dtype=object)
        for row in range(rows):
            method_axis = fig.add_subplot(grid[row, 0])
            _draw_method_cell(method_axis, row, stage_sets[0][row][0])
            for col in range(columns):
                axes[row][col] = fig.add_subplot(grid[row, col + 1])
    else:
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(fig_width, fig_height),
            squeeze=False,
            facecolor="white",
        )

    for col, stage_images in enumerate(stage_sets):
        for row, (stage_name, image) in enumerate(stage_images):
            axis = axes[row][col]
            display_image = image
            if paper_style:
                background = (0, 0, 0) if stage_name == "Segmentation Result" else (255, 255, 255)
                display_image = _fit_image_to_canvas(image, background=background)
            axis.imshow(display_image)
            axis.axis("off")
            if label_each_panel:
                axis.text(
                    0.02,
                    0.96,
                    stage_name,
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="white",
                    bbox={
                        "facecolor": "black",
                        "alpha": 0.65,
                        "edgecolor": "none",
                        "boxstyle": "round,pad=0.25",
                    },
                )
            if col == 0 and not (paper_style and show_method_column):
                axis.set_ylabel(stage_name, rotation=0, ha="right", va="center", labelpad=55)
            if row == 0:
                column_title = (
                    column_titles[col]
                    if column_titles and col < len(column_titles)
                    else f"Image {col + 1}"
                )
                axis.set_title(column_title)

    if title:
        fig.text(0.5, 0.02, title, ha="center", va="bottom", fontsize=9)

    if not (paper_style and show_method_column):
        plt.tight_layout()
    return fig


def show_workflow_grid(
    results,
    column_titles=None,
    figsize_per_cell=(3.0, 2.2),
    label_each_panel: bool = True,
    paper_style: bool = True,
    show_method_column: bool = True,
    title: str | None = "Workflow and result for building facade RGB images parsing.",
    segmentation_view: str = "overlay",
):
    """Display results like a stage-by-stage workflow figure.

    Rows are stages; columns are analysed images.
    """

    build_workflow_grid_figure(
        results,
        column_titles,
        figsize_per_cell,
        label_each_panel=label_each_panel,
        paper_style=paper_style,
        show_method_column=show_method_column,
        title=title,
        segmentation_view=segmentation_view,
    )
    plt.show()


def save_workflow_grid_image(
    results,
    path: str,
    column_titles=None,
    figsize_per_cell=(3.0, 2.2),
    dpi: int = 300,
    label_each_panel: bool = True,
    paper_style: bool = True,
    show_method_column: bool = True,
    title: str | None = "Workflow and result for building facade RGB images parsing.",
    segmentation_view: str = "overlay",
):
    """Save workflow grid to PNG/JPG.

    JPG is written through OpenCV to avoid occasional Colab/Pillow savefig
    incompatibilities.
    """

    fig = build_workflow_grid_figure(
        results,
        column_titles,
        figsize_per_cell,
        label_each_panel=label_each_panel,
        paper_style=paper_style,
        show_method_column=show_method_column,
        title=title,
        segmentation_view=segmentation_view,
    )
    path_lower = path.lower()

    if path_lower.endswith((".jpg", ".jpeg")):
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.asarray(fig.canvas.buffer_rgba()).reshape((height, width, 4))
        image_rgb = rgba[:, :, :3].copy()
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(path, image_bgr)
        if not ok:
            raise OSError(f"Could not save workflow grid to {path}")
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    return path


def _methodology_stage_images(result):
    """Return the five stage images used by the circular methodology figure."""

    image_rgb = result["image_rgb"]
    obstacle_mask = result.get(
        "obstacle_mask",
        np.zeros(image_rgb.shape[:2], dtype=bool),
    )
    clean_image = result.get("clean_image", image_rgb)
    aligned_facade = result.get("aligned_facade", clean_image)

    obstacle_detection = make_mask_overlay(
        image_rgb,
        obstacle_mask.astype(bool),
        color=(255, 0, 0),
        alpha=0.52,
    )

    segmentation_image = make_binary_mask_image(result["usable_results"]["usable_mask"])

    return {
        1: image_rgb,
        2: obstacle_detection,
        3: clean_image,
        4: aligned_facade,
        5: segmentation_image,
    }


def _add_methodology_card(axis, center, size, number, title, body, image, edge_color):
    """Draw one rounded methodology stage card."""

    from matplotlib.patches import Circle, FancyBboxPatch

    x, y = center
    width, height = size
    left = x - width / 2
    bottom = y - height / 2

    card = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.7,
        edgecolor=edge_color,
        facecolor="white",
        zorder=3,
    )
    axis.add_patch(card)

    circle = Circle(
        (left + 0.035, bottom + height - 0.005),
        0.026,
        facecolor=edge_color,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )
    axis.add_patch(circle)
    axis.text(
        left + 0.035,
        bottom + height - 0.005,
        str(number),
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        weight="bold",
        zorder=6,
    )

    axis.text(
        left + 0.016,
        bottom + height - 0.032,
        title,
        ha="left",
        va="top",
        fontsize=10,
        color="#17324d",
        weight="bold",
        zorder=6,
    )
    axis.text(
        left + 0.016,
        bottom + height - 0.072,
        body,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#222222",
        linespacing=1.15,
        zorder=6,
    )

    thumbnail = _fit_image_to_canvas(image, canvas_size=(130, 95), background=(245, 247, 248))
    img_w = width * 0.34
    img_h = height * 0.58
    img_left = left + width - img_w - 0.018
    img_bottom = bottom + 0.028
    axis.imshow(
        thumbnail,
        extent=[img_left, img_left + img_w, img_bottom, img_bottom + img_h],
        zorder=6,
    )


def build_methodology_overview_figure(
    result,
    figsize=(11, 11),
    title="IMAGE-BASED PIPELINE\nFOR BIPV FACADE AREA\nESTIMATION",
):
    """Build a five-stage circular methodology overview for one analysed image."""

    from matplotlib.patches import Circle, FancyArrowPatch

    stage_images = _methodology_stage_images(result)
    fig, axis = plt.subplots(figsize=figsize, facecolor="white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    positions = {
        1: (0.50, 0.86),
        2: (0.81, 0.62),
        3: (0.70, 0.23),
        4: (0.30, 0.23),
        5: (0.19, 0.62),
    }
    colors = {
        1: "#1f4e8c",
        2: "#2a9bb0",
        3: "#6aa84f",
        4: "#f28c28",
        5: "#8e44ad",
    }
    stage_text = {
        1: (
            "FACADE IMAGE",
            "Street-level facade\nimage used as the\npipeline input.",
        ),
        2: (
            "OBSTACLE DETECTION",
            "Grounding DINO detects\nforeground obstructions\non the facade image.",
        ),
        3: (
            "OBSTACLE REMOVAL",
            "SAM and inpainting\nremove detected\nobstacles.",
        ),
        4: (
            "FACADE ALIGNMENT",
            "Perspective distortion\nis corrected for a\nfront-facing facade.",
        ),
        5: (
            "SEGMENTATION RESULT",
            "Facade wall and excluded\nopenings are converted\ninto the final mask.",
        ),
    }

    ring_points = [
        (0.50, 0.76),
        (0.70, 0.68),
        (0.76, 0.34),
        (0.50, 0.20),
        (0.24, 0.34),
        (0.30, 0.68),
        (0.50, 0.76),
    ]
    arrow_colors = ["#2f6fa8", "#2f8f8d", "#59a14f", "#f28c28", "#8e44ad", "#5b6fb0"]
    for start, end, arrow_color in zip(ring_points[:-1], ring_points[1:], arrow_colors):
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=4.0,
                color=arrow_color,
                alpha=0.82,
                connectionstyle="arc3,rad=0.08",
                zorder=1,
            )
        )

    center_circle = Circle(
        (0.5, 0.505),
        0.17,
        facecolor="white",
        edgecolor="#e2e2e2",
        linewidth=1.2,
        zorder=2,
    )
    axis.add_patch(center_circle)
    axis.text(
        0.5,
        0.56,
        title,
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
        color="#153a63",
        linespacing=1.15,
        zorder=4,
    )
    axis.text(
        0.5,
        0.485,
        "From Street-Level Image to\nPVsyst-Ready Data",
        ha="center",
        va="center",
        fontsize=10,
        color="#35495e",
        zorder=4,
    )
    axis.plot([0.39, 0.61], [0.435, 0.435], color="#cccccc", linewidth=1.0, zorder=4)
    axis.text(
        0.5,
        0.405,
        "FINAL OUTPUTS",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color="#153a63",
        zorder=4,
    )
    axis.text(
        0.5,
        0.36,
        "Usable BIPV Area   Facade Segmentation   Excel   JSON",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#2f6f3e",
        zorder=4,
    )

    card_size = (0.29, 0.155)
    for number, center in positions.items():
        stage_title, body = stage_text[number]
        _add_methodology_card(
            axis,
            center,
            card_size,
            number,
            stage_title,
            body,
            stage_images[number],
            colors[number],
        )

    plt.tight_layout(pad=0.2)
    return fig


def show_methodology_overview(result, figsize=(11, 11)):
    """Display a circular methodology overview for one analysed image."""

    build_methodology_overview_figure(result, figsize=figsize)
    plt.show()


def save_methodology_overview_image(result, path: str, figsize=(11, 11), dpi: int = 300):
    """Save a circular methodology overview figure to PNG/JPG."""

    fig = build_methodology_overview_figure(result, figsize=figsize)
    path_lower = path.lower()

    if path_lower.endswith((".jpg", ".jpeg")):
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.asarray(fig.canvas.buffer_rgba()).reshape((height, width, 4))
        image_rgb = rgba[:, :, :3].copy()
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(path, image_bgr)
        if not ok:
            raise OSError(f"Could not save methodology overview to {path}")
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    return path


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

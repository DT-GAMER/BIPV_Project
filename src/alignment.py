"""Stage 6: facade alignment and grid structuring."""

from __future__ import annotations

import numpy as np


def infer_floor_bands(window_boxes_np, min_floor_gap: float = 0.06):
    """Group normalized window boxes into horizontal floor bands."""

    if len(window_boxes_np) == 0:
        return []

    centers_y = np.sort(window_boxes_np[:, 1])
    bands = [[float(centers_y[0])]]
    for center_y in centers_y[1:]:
        if center_y - bands[-1][-1] > min_floor_gap:
            bands.append([])
        bands[-1].append(float(center_y))

    return [
        {
            "center_y": float(np.mean(band)),
            "count": len(band),
        }
        for band in bands
    ]


def infer_window_columns(window_boxes_np, tolerance_x: float = 0.035):
    """Group normalized window boxes into vertical columns."""

    if len(window_boxes_np) == 0:
        return []

    centers_x = sorted(float(box[0]) for box in window_boxes_np)
    columns = [[centers_x[0]]]
    for center_x in centers_x[1:]:
        if abs(center_x - np.mean(columns[-1])) <= tolerance_x:
            columns[-1].append(center_x)
        else:
            columns.append([center_x])

    return [
        {
            "center_x": float(np.mean(column)),
            "count": len(column),
        }
        for column in columns
    ]


def align_facade_grid(window_boxes_np):
    """Return structured facade grid metadata from detected windows."""

    return {
        "floors": infer_floor_bands(window_boxes_np),
        "columns": infer_window_columns(window_boxes_np),
    }

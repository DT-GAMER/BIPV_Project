"""Usable facade area and BIPV capacity calculations."""

from __future__ import annotations

import numpy as np

from .utils import dilate_mask


def mask_extent(mask):
    """Return height and width of the non-zero mask bounding box."""

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask.shape[:2]
    return ys.max() - ys.min() + 1, xs.max() - xs.min() + 1


def count_floors_from_windows(window_boxes_np, min_floor_gap: float = 0.06) -> int:
    centers_y = np.sort(window_boxes_np[:, 1])
    if len(centers_y) < 2:
        return len(centers_y) if len(centers_y) > 0 else 5
    gaps = np.diff(centers_y)
    return int(np.clip(int(np.sum(gaps > min_floor_gap)) + 1, 2, 20))


def calculate_real_world_dimensions(
    aligned_facade,
    window_boxes_np,
    known_floors: int | None = None,
    reference_height_m: float = 3.0,
    validated_width_m: float | None = None,
    validated_height_m: float | None = None,
    facade_mask=None,
):
    if facade_mask is None:
        height_px, width_px = aligned_facade.shape[:2]
    else:
        height_px, width_px = mask_extent(facade_mask)

    if known_floors is not None:
        num_floors = int(known_floors)
    elif len(window_boxes_np) > 0:
        num_floors = count_floors_from_windows(window_boxes_np)
    else:
        num_floors = 5

    height_m = validated_height_m if validated_height_m else num_floors * reference_height_m
    pixels_per_meter = height_px / height_m
    width_m = validated_width_m if validated_width_m else width_px / pixels_per_meter

    return {
        "num_floors": num_floors,
        "height_m": height_m,
        "width_m": width_m,
        "pixels_per_meter": pixels_per_meter,
        "total_facade_area_m2": height_m * width_m,
    }


def calculate_usable_area(
    facade_mask,
    window_mask,
    door_mask,
    balcony_mask,
    shadow_mask,
    dimensions,
):
    height, width = facade_mask.shape
    usable_mask = facade_mask.copy()
    usable_mask &= ~dilate_mask(window_mask, kernel_size=5, iterations=1)
    usable_mask &= ~dilate_mask(door_mask, kernel_size=5, iterations=1)
    usable_mask &= ~dilate_mask(balcony_mask, kernel_size=7, iterations=1)

    usable_no_shadow = usable_mask.copy()
    heavily_shadowed = shadow_mask & facade_mask
    usable_reduced = usable_mask & ~heavily_shadowed

    facade_area_px = facade_mask.sum()
    if facade_area_px == 0:
        px_to_m2 = 0
    else:
        px_to_m2 = dimensions["total_facade_area_m2"] / facade_area_px

    return {
        "facade_area_m2": dimensions["total_facade_area_m2"] if facade_area_px else 0,
        "usable_area_m2": usable_no_shadow.sum() * px_to_m2,
        "usable_area_reduced_m2": usable_reduced.sum() * px_to_m2,
        "window_area_m2": window_mask.sum() * px_to_m2,
        "door_area_m2": door_mask.sum() * px_to_m2,
        "balcony_area_m2": balcony_mask.sum() * px_to_m2,
        "shadow_area_m2": heavily_shadowed.sum() * px_to_m2,
        "usable_percentage": 100 * usable_no_shadow.sum() / facade_area_px if facade_area_px > 0 else 0,
        "px_to_m2": px_to_m2,
        "facade_area_px": int(facade_area_px),
        "usable_mask": usable_no_shadow,
        "usable_mask_reduced": usable_reduced,
    }


def estimate_panel_capacity(
    usable_area_m2: float,
    panel_efficiency: float = 0.20,
    panel_area_m2: float = 1.7,
    watts_per_panel: int = 350,
):
    num_panels = int(usable_area_m2 / panel_area_m2)
    return {
        "num_panels": num_panels,
        "total_capacity_kw": (num_panels * watts_per_panel) / 1000,
        "panel_efficiency": panel_efficiency,
        "panel_area_m2": panel_area_m2,
        "watts_per_panel": watts_per_panel,
    }

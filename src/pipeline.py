"""End-to-end BIPV facade analysis pipeline."""

from __future__ import annotations

import gc
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from .alignment import align_facade_grid
from .area import calculate_usable_area
from .bipv_segmentation import segment_bipv_surface, warp_mask
from .config import AnalysisConfig
from .detection import annotate, detect_obstacles_and_architecture
from .energy import estimate_energy_yield, estimate_panel_capacity
from .export import (
    excel_path_from_json_path,
    prepare_pvsyst_export,
    save_pvsyst_excel,
    save_pvsyst_export,
)
from .geometry import (
    building_bbox_from_boxes,
    rectify_facade,
)
from .inpainting import (
    build_obstacle_box_mask,
    build_robust_mask,
    remove_obstacles,
    segment_obstacles_with_sam,
)
from .model_loader import load_models
from .preprocessing import load_and_preprocess_image
from .scaling import estimate_real_world_scale
from .segmentation import _clean_facade_boundary, segment_facade_components
from .trained_facade_parser import run_trained_facade_parser
from .utils import dilate_mask


def _disabled_shadow_analysis(facade_mask):
    """Return a neutral shadow result while shadow analysis is out of scope."""

    shadow_mask = np.zeros_like(facade_mask, dtype=bool)
    return {
        "shadow_area_px": 0,
        "shadow_percentage": 0.0,
        "shadow_mask": shadow_mask,
        "status": "disabled",
    }


def _resolve_trained_facade_parser_path(config: AnalysisConfig) -> Path | None:
    """Return the first trained facade parser weights file available."""

    candidates = []
    if config.trained_facade_parser_path:
        candidates.append(Path(config.trained_facade_parser_path))
    candidates.append(Path("models/facade_parser.pt"))
    if config.trained_facade_parser_drive_path:
        candidates.append(Path(config.trained_facade_parser_drive_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _measurement_quality_from_trained_parser(quality, facade_mask, window_mask):
    """Create an engineering quality flag for trained parser output."""

    height, width = facade_mask.shape[:2]
    facade_coverage = float(facade_mask.sum() / max(height * width, 1) * 100)
    window_components = _window_boxes_from_mask(window_mask, facade_mask)
    issues = []

    if quality.get("status") != "ok":
        issues.append(f"trained-parser-status:{quality.get('status')}")
    if facade_coverage < 8:
        issues.append("facade-mask-too-small")
    if facade_coverage > 85:
        issues.append("facade-mask-too-large")
    if len(window_components) < 4:
        issues.append("few-trained-window-detections")

    if not issues:
        return {
            "status": "calculation_ready",
            "confidence": 0.90,
            "issues": [],
            "message": "Trained facade parser produced usable facade/opening masks.",
        }
    if len(issues) <= 2 and len(window_components) >= 4:
        return {
            "status": "review_recommended",
            "confidence": 0.70,
            "issues": issues,
            "message": "Trained segmentation ran, but the result should be visually checked.",
        }
    return {
        "status": "manual_review_required",
        "confidence": 0.40,
        "issues": issues,
        "message": "Trained segmentation is not reliable enough for final engineering use without review.",
    }


def _window_boxes_from_mask(window_mask, facade_mask):
    """Extract normalized xywh boxes from a binary opening mask."""

    height, width = window_mask.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        window_mask.astype(np.uint8),
        connectivity=8,
    )
    facade_area = max(int(facade_mask.sum()), 1)
    boxes = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        area_fraction = float(area / facade_area)
        if area_fraction < 0.00015 or area_fraction > 0.08:
            continue
        if w < 3 or h < 3:
            continue
        boxes.append(
            [
                float((x + w / 2) / max(width, 1)),
                float((y + h / 2) / max(height, 1)),
                float(w / max(width, 1)),
                float(h / max(height, 1)),
            ]
        )
    return np.array(boxes, dtype=float)


def _segmentation_from_trained_parser(aligned_facade, config, device):
    """Run the trained parser when weights exist; otherwise return None."""

    weights_path = _resolve_trained_facade_parser_path(config)
    if not config.use_trained_facade_parser:
        return None, {
            "enabled": False,
            "status": "not-used",
            "reason": "disabled-in-config",
        }
    if weights_path is None:
        return None, {
            "enabled": config.use_trained_facade_parser,
            "status": "not-used",
            "reason": "no-trained-weights-found",
        }

    try:
        parser_result = run_trained_facade_parser(
            aligned_facade,
            str(weights_path),
            conf=config.trained_facade_parser_conf,
            imgsz=config.trained_facade_parser_imgsz,
            device=device,
        )
    except Exception as exc:
        return None, {
            "enabled": True,
            "status": "fallback",
            "reason": f"trained-parser-error:{type(exc).__name__}",
            "message": str(exc),
            "weights_path": str(weights_path),
        }

    facade_mask = parser_result.facade_mask.astype(bool)
    if facade_mask.sum() == 0:
        # A small first dataset may learn windows before it learns the full
        # facade wall. Keep the opening masks available so the main pipeline can
        # clip them to the stronger DINO/SAM facade geometry mask.
        facade_mask = np.ones(parser_result.window_mask.shape, dtype=bool)
    window_mask = (parser_result.window_mask & facade_mask).astype(bool)
    door_mask = (parser_result.door_mask & facade_mask).astype(bool)
    balcony_mask = (parser_result.balcony_mask & facade_mask).astype(bool)
    roof_mask = (parser_result.roof_edge_mask & facade_mask).astype(bool)
    facade_coverage = float(facade_mask.sum() / max(facade_mask.size, 1) * 100)
    window_boxes = _window_boxes_from_mask(window_mask, facade_mask)
    quality = {
        **parser_result.quality,
        "parser_source": "trained_facade_parser",
        "trained_facade_parser_path": str(weights_path),
        "trained_window_count": int(len(window_boxes)),
        "facade_coverage_percent": facade_coverage,
    }
    quality["measurement_quality"] = _measurement_quality_from_trained_parser(
        parser_result.quality,
        facade_mask,
        window_mask,
    )

    return {
        "facade_mask": facade_mask,
        "window_mask": window_mask,
        "raw_window_mask": window_mask.copy(),
        "door_mask": door_mask,
        "balcony_mask": balcony_mask,
        "roof_mask": roof_mask,
        "boxes": torch.empty((0, 4)),
        "logits": torch.empty((0,)),
        "phrases": [],
        "auto_masks": [],
        "quality": quality,
    }, {
        "enabled": True,
        "status": "used",
        "weights_path": str(weights_path),
    }


def _merge_trained_openings_into_segmentation(base_segmentation, trained_segmentation):
    """Use the trained parser for openings while preserving base facade geometry."""

    merged = dict(base_segmentation)
    facade_mask = base_segmentation["facade_mask"].astype(bool)
    base_window_mask = (base_segmentation["window_mask"] & facade_mask).astype(bool)
    trained_window_mask = (trained_segmentation["window_mask"] & facade_mask).astype(bool)

    base_window_count = len(_window_boxes_from_mask(base_window_mask, facade_mask))
    trained_window_count = len(_window_boxes_from_mask(trained_window_mask, facade_mask))
    use_trained_windows = trained_window_count >= max(4, int(base_window_count * 0.45))

    if use_trained_windows:
        window_mask = trained_window_mask
        window_source = "trained_facade_parser"
    else:
        window_mask = base_window_mask
        window_source = "dino_sam_fallback"

    door_mask = (
        base_segmentation["door_mask"]
        | (trained_segmentation["door_mask"] & facade_mask)
    ).astype(bool)
    balcony_mask = (
        base_segmentation["balcony_mask"]
        | (trained_segmentation["balcony_mask"] & facade_mask)
    ).astype(bool)
    roof_mask = (trained_segmentation.get("roof_mask", np.zeros_like(facade_mask)) & facade_mask).astype(bool)

    quality = dict(base_segmentation["quality"])
    quality.update(
        {
            "parser_source": "hybrid_dino_sam_facade_trained_openings",
            "trained_openings_applied": bool(use_trained_windows),
            "window_mask_source": window_source,
            "base_window_count": int(base_window_count),
            "trained_window_count": int(trained_window_count),
            "trained_roof_edge_pixels": int(roof_mask.sum()),
            "trained_facade_parser_path": trained_segmentation["quality"].get(
                "trained_facade_parser_path"
            ),
        }
    )
    if use_trained_windows:
        quality["measurement_quality"] = _measurement_quality_from_trained_parser(
            {"status": "ok"},
            facade_mask,
            window_mask,
        )

    merged.update(
        {
            "facade_mask": facade_mask,
            "window_mask": window_mask,
            "raw_window_mask": window_mask.copy(),
            "door_mask": door_mask & facade_mask,
            "balcony_mask": balcony_mask & facade_mask,
            "roof_mask": roof_mask,
            # Force downstream floor/scale extraction to use the final opening
            # mask rather than stale DINO boxes when trained openings are used.
            "boxes": torch.empty((0, 4)),
            "logits": torch.empty((0,)),
            "phrases": [],
            "quality": quality,
        }
    )
    return merged


def _remove_mask_components(mask, remove_region=None, top_fraction: float | None = None):
    """Remove implausible exclusion components from a binary mask."""

    if mask.sum() == 0:
        return mask

    height, width = mask.shape[:2]
    cleaned = mask.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        component = labels == label_id
        remove = False
        if remove_region is not None:
            overlap = int((component & remove_region).sum()) / max(int(area), 1)
            remove = overlap > 0.12
        if top_fraction is not None and y <= int(height * top_fraction):
            # Top-edge door/balcony detections are usually roof/parapet false
            # positives in this workflow, not usable-area exclusions.
            remove = True
        if remove:
            cleaned[component] = False

    return cleaned


def _clean_architectural_exclusions(segmentation, reconstructed_mask, config):
    """Reduce false orange exclusions caused by inpainted obstacle regions."""

    facade_mask = segmentation["facade_mask"]
    cleanup_region = None
    if config.suppress_architecture_on_reconstructed_regions and reconstructed_mask is not None:
        cleanup_region = dilate_mask(
            reconstructed_mask & facade_mask,
            kernel_size=11,
            iterations=1,
        )

    segmentation["door_mask"] = _remove_mask_components(
        segmentation["door_mask"] & facade_mask,
        remove_region=cleanup_region,
        top_fraction=0.45,
    )
    segmentation["balcony_mask"] = _remove_mask_components(
        segmentation["balcony_mask"] & facade_mask,
        remove_region=cleanup_region,
        top_fraction=0.06,
    )
    if "roof_mask" in segmentation:
        segmentation["roof_mask"] = _remove_mask_components(
            segmentation["roof_mask"] & facade_mask,
            remove_region=cleanup_region,
            top_fraction=None,
        )

    quality = dict(segmentation.get("quality", {}))
    quality["architectural_exclusion_cleanup"] = {
        "reconstructed_region_suppression": bool(cleanup_region is not None),
        "door_pixels_after_cleanup": int(segmentation["door_mask"].sum()),
        "balcony_pixels_after_cleanup": int(segmentation["balcony_mask"].sum()),
        "roof_pixels_after_cleanup": int(segmentation.get("roof_mask", np.zeros_like(facade_mask)).sum()),
    }
    segmentation["quality"] = quality
    return segmentation


def _set_random_seeds(seed: int) -> None:
    """Lock all RNGs so SAM point-sampling and CUDA ops give the same result every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _exclude_ground_floor_from_mask(
    segmentation: dict, dimensions: dict
) -> tuple[dict, dict]:
    """Remove the ground floor from the facade mask and update area dimensions.

    The ground floor (commercial units, entrances, pavement level) is excluded
    from BIPV assessment because it is not accessible for panel installation.
    One floor's worth of height is trimmed from the bottom of the facade mask.
    Dimensions are updated proportionally so px_to_m2 stays consistent.
    """
    num_floors = dimensions.get("num_floors", 1)
    if num_floors < 2:
        return segmentation, dimensions

    facade_mask = segmentation["facade_mask"]
    ys, _ = np.where(facade_mask)
    if len(ys) == 0:
        return segmentation, dimensions

    facade_bottom_y = int(ys.max())
    facade_top_y = int(ys.min())
    facade_height_px = facade_bottom_y - facade_top_y + 1
    ground_floor_px = int(round(facade_height_px / num_floors))
    cutoff_y = facade_bottom_y - ground_floor_px + 1

    ground_floor_zone = np.zeros_like(facade_mask, dtype=bool)
    if cutoff_y < facade_mask.shape[0]:
        ground_floor_zone[cutoff_y:, :] = True

    updated_facade = facade_mask & ~ground_floor_zone
    segmentation["facade_mask"] = updated_facade
    for key in ("window_mask", "raw_window_mask", "door_mask", "balcony_mask", "roof_mask"):
        if key in segmentation:
            segmentation[key] = segmentation[key] & updated_facade

    new_num_floors = num_floors - 1
    new_height_m = dimensions["height_m"] * new_num_floors / num_floors
    updated_dims = dict(dimensions)
    updated_dims["num_floors"] = new_num_floors
    updated_dims["height_m"] = new_height_m
    updated_dims["total_facade_area_m2"] = new_height_m * dimensions["width_m"]
    updated_dims["ground_floor_excluded"] = True
    updated_dims["ground_floor_height_m"] = round(dimensions["height_m"] / num_floors, 2)
    return segmentation, updated_dims


def _remove_top_roof_pixels(facade_mask: np.ndarray, aligned_facade: np.ndarray) -> np.ndarray:
    """Remove roof-like pixels from the upper facade mask.

    This is intentionally conservative and only acts in the top portion of the
    mask. It targets detached/low-rise house cases where dark pitched roof
    surfaces leak into the facade mask while leaving the wall/gable regions.
    """

    if facade_mask.sum() == 0:
        return facade_mask.astype(bool)

    height, _ = facade_mask.shape
    ys, _ = np.where(facade_mask)
    top = int(ys.min())
    bottom = int(ys.max())
    facade_height = max(bottom - top + 1, 1)
    top_limit = min(height, top + int(round(facade_height * 0.38)))

    top_zone = np.zeros_like(facade_mask, dtype=bool)
    top_zone[top:top_limit, :] = True

    hsv = cv2.cvtColor(aligned_facade, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(aligned_facade, cv2.COLOR_RGB2GRAY)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Pitched roofs are commonly dark grey/blue panels or dark slate/tile in the
    # top facade band. White/light gable wall is intentionally not removed.
    dark_roof = (
        ((value < 155) & (saturation < 95))
        | ((hue >= 88) & (hue <= 135) & (saturation > 35) & (value < 185))
        | (gray < 115)
    )
    candidate = dark_roof & facade_mask.astype(bool) & top_zone

    cleaned = np.zeros_like(candidate, dtype=bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8),
        connectivity=8,
    )
    facade_area = int(facade_mask.sum())
    for label_id in range(1, num_labels):
        x, y, component_w, component_h, area = stats[label_id]
        if area < max(20, facade_area * 0.002):
            continue
        if y > top + facade_height * 0.30:
            continue
        if component_h > facade_height * 0.32:
            continue
        if component_w < 8:
            continue
        cleaned |= labels == label_id

    # Avoid destructive removals on dark-stone buildings.
    if cleaned.sum() > facade_area * 0.18:
        return facade_mask.astype(bool)

    return facade_mask.astype(bool) & ~cleaned


def _regularize_small_opening_components(
    opening_mask: np.ndarray,
    facade_mask: np.ndarray,
) -> np.ndarray:
    """Make small low-rise window/opening components cleaner rectangles."""

    if opening_mask.sum() == 0 or facade_mask.sum() == 0:
        return opening_mask.astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opening_mask.astype(np.uint8),
        connectivity=8,
    )
    component_count = num_labels - 1
    if component_count == 0 or component_count > 22:
        return opening_mask.astype(bool)

    facade_area = int(facade_mask.sum())
    regularized = np.zeros_like(opening_mask, dtype=bool)
    changed = False

    for label_id in range(1, num_labels):
        x, y, component_w, component_h, area = stats[label_id]
        if area <= 0 or component_w <= 2 or component_h <= 2:
            continue

        bbox_area = component_w * component_h
        area_fraction = bbox_area / max(facade_area, 1)
        aspect = component_h / max(component_w, 1)
        fill = area / max(bbox_area, 1)

        if 0.0003 <= area_fraction <= 0.06 and 0.25 <= aspect <= 5.5 and fill >= 0.20:
            pad_x = max(0, int(round(component_w * 0.04)))
            pad_y = max(0, int(round(component_h * 0.04)))
            candidate = np.zeros_like(opening_mask, dtype=bool)
            candidate[y + pad_y : y + component_h - pad_y, x + pad_x : x + component_w - pad_x] = True
            if (candidate & facade_mask).sum() / max(candidate.sum(), 1) >= 0.65:
                regularized |= candidate & facade_mask
                changed = True
                continue

        regularized |= labels == label_id

    if not changed:
        return opening_mask.astype(bool)

    # Keep the correction bounded; if rectangles massively change the area, use
    # the original mask because it is safer for area estimation.
    ratio = regularized.sum() / max(opening_mask.sum(), 1)
    if not 0.65 <= ratio <= 1.85:
        return opening_mask.astype(bool)

    return regularized & facade_mask


def _postprocess_facade_mask(segmentation: dict, aligned_facade: np.ndarray) -> dict:
    """Remove sky pixels and smooth facade mask edges after segmentation."""

    facade_mask = segmentation["facade_mask"].astype(np.uint8)
    height = facade_mask.shape[0]

    # Sky exclusion: only search the top 55% of the image where sky actually is.
    hsv = cv2.cvtColor(aligned_facade, cv2.COLOR_RGB2HSV)
    sky_zone = np.zeros(facade_mask.shape, dtype=bool)
    sky_zone[: int(height * 0.55), :] = True
    blue_sky = (
        (hsv[:, :, 0] >= 88) & (hsv[:, :, 0] <= 138)
        & (hsv[:, :, 1] >= 45)
        & (hsv[:, :, 2] >= 90)
    )
    overcast_sky = (hsv[:, :, 1] < 28) & (hsv[:, :, 2] >= 210)
    sky_pixels = (blue_sky | overcast_sky) & sky_zone
    # Sky removal
    facade_after_sky = facade_mask.astype(bool) & ~sky_pixels

    # Re-apply clean boundary after all pipeline AND-clips (rectified_content_mask,
    # architectural exclusions) which re-introduce jagged SAM edges.
    facade_clean = _clean_facade_boundary(facade_after_sky)
    facade_clean = _remove_top_roof_pixels(facade_clean, aligned_facade)

    # Re-clip all derived masks to the final clean boundary.
    segmentation["facade_mask"] = facade_clean
    for key in ("window_mask", "raw_window_mask", "door_mask", "balcony_mask", "roof_mask"):
        if key in segmentation:
            segmentation[key] = segmentation[key] & facade_clean

    for key in ("window_mask", "raw_window_mask"):
        if key in segmentation:
            segmentation[key] = _regularize_small_opening_components(
                segmentation[key],
                facade_clean,
            )

    return segmentation


def run_bipv_analysis(config: AnalysisConfig | None = None, models=None, **kwargs):
    """Run the full BIPV analysis.

    Pass either an ``AnalysisConfig`` or keyword arguments accepted by it.
    This function is intended for Google Colab GPU execution.
    """

    config = config or AnalysisConfig(**kwargs)
    _set_random_seeds(config.random_seed)
    owns_models = models is None
    models = models or load_models(load_stable_diffusion=config.run_stable_diffusion)
    device = models["device"]

    print("Stage 1/8 - Image acquisition and preprocessing")
    image_rgb = load_and_preprocess_image(config.image_path, max_side=config.max_image_side)
    stages = {}
    stages["preprocessing"] = {
        "image_shape": tuple(image_rgb.shape),
        "max_image_side": config.max_image_side,
    }

    print("Stage 2/8 - Facade object detection")
    source_detection = detect_obstacles_and_architecture(
        image_rgb,
        models["dino_model"],
        device,
        facade_roi_bottom=config.facade_roi_bottom,
        box_threshold=config.dino_box_threshold,
        text_threshold=config.dino_text_threshold,
    )
    height, width = image_rgb.shape[:2]
    bx1, by1, bx2, by2, keep_boxes, facade_selection_quality = building_bbox_from_boxes(
        source_detection.boxes,
        source_detection.keep_ids,
        height,
        width,
        facade_roi_bottom=config.facade_roi_bottom,
    )
    facade_constraint_mask = np.zeros((height, width), dtype=bool)
    if config.constrain_obstacles_to_facade:
        pad_x = int(width * 0.05)
        pad_y = int(height * 0.02)
        fx1 = max(0, int(bx1) - pad_x)
        fy1 = max(0, int(by1) - pad_y)
        fx2 = min(width - 1, int(bx2) + pad_x)
        fy2 = height - 1
        facade_constraint_mask[fy1:fy2, fx1:fx2] = True
    else:
        facade_constraint_mask[:, :] = True
    stages["source_detection"] = {
        "detections": len(source_detection.phrases),
        "architectural": len(source_detection.keep_ids),
        "selected_architectural": len(keep_boxes),
        "obstacles": len(source_detection.remove_ids),
        "source_building_bbox": [float(bx1), float(by1), float(bx2), float(by2)],
        **facade_selection_quality,
    }

    print("Stage 3/8 - Image segmentation")
    raw_obstacle_mask = segment_obstacles_with_sam(
        image_rgb,
        source_detection.boxes,
        source_detection.remove_ids,
        models["predictor"],
    )
    obstacle_box_mask = build_obstacle_box_mask(
        image_rgb.shape,
        source_detection.boxes,
        source_detection.remove_ids,
        source_detection.phrases,
    )
    raw_obstacle_mask |= obstacle_box_mask
    raw_obstacle_mask &= facade_constraint_mask
    print("Stage 4/8 - Obstacle masking")
    robust_mask = build_robust_mask(
        raw_obstacle_mask,
        dilate_kernel=config.obstacle_dilate_kernel,
        dilate_iters=config.obstacle_dilate_iters,
        shadow_pad_frac=config.obstacle_shadow_pad_frac,
        max_mask_fraction=config.max_obstacle_mask_fraction,
    )
    robust_mask &= facade_constraint_mask
    stages["obstacle_segmentation"] = {
        "raw_obstacle_pixels": int(raw_obstacle_mask.sum()),
        "obstacle_box_pixels": int((obstacle_box_mask & facade_constraint_mask).sum()),
        "robust_obstacle_pixels": int(robust_mask.sum()),
    }

    print("Stage 5/8 - Obstacle removal / inpainting")
    clean_image = remove_obstacles(
        image_rgb,
        robust_mask,
        models["lama"],
        sd_pipe=models.get("sd_pipe"),
        run_stable_diffusion=config.run_stable_diffusion,
    )
    if models.get("sd_pipe") is not None:
        del models["sd_pipe"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if clean_image.shape[:2] != image_rgb.shape[:2]:
        raise ValueError(
            "Obstacle removal changed the image size. This would break metric area calculation."
        )
    stages["obstacle_removal"] = {
        "obstacle_pixels": int(robust_mask.sum()),
        "obstacle_mask_fraction": float(robust_mask.sum() / max(height * width, 1)),
        "image_shape": tuple(clean_image.shape),
        "stable_diffusion_used": config.run_stable_diffusion,
        "constrained_to_facade": config.constrain_obstacles_to_facade,
    }

    print("Stage 6/8 - Facade rectification")
    # Run SAM on the obstacle-removed image to find the actual facade polygon.
    # This gives proper 4-corner perspective correction instead of the axis-aligned
    # bounding-box approach which produces near-identity transforms.
    from .geometry import find_facade_quad_from_mask
    h_img, w_img = clean_image.shape[:2]
    bbox_region = np.zeros((h_img, w_img), dtype=bool)
    bbox_region[
        max(0, int(by1)) : min(h_img, int(by2) + 1),
        max(0, int(bx1)) : min(w_img, int(bx2) + 1),
    ] = True
    quick_auto_masks = models["mask_generator"].generate(clean_image)
    facade_quad = None
    for mask_data in sorted(quick_auto_masks, key=lambda m: m["area"], reverse=True)[:10]:
        candidate = mask_data["segmentation"] & bbox_region
        coverage = float(candidate.sum()) / max(float(bbox_region.sum()), 1)
        if not (0.30 <= coverage <= 1.05):
            continue
        ys_c, xs_c = np.where(candidate)
        if len(xs_c) == 0:
            continue
        c_h = int(ys_c.max() - ys_c.min() + 1)
        c_w = int(xs_c.max() - xs_c.min() + 1)
        if candidate.sum() / max(c_h * c_w, 1) < 0.38:
            continue
        corners = find_facade_quad_from_mask(candidate)
        if corners is not None:
            facade_quad = corners
            break
    del quick_auto_masks  # free memory before full segmentation SAM run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Metric area estimates require rectification to keep the original image
    # canvas and detected building footprint stable.
    rectification = rectify_facade(
        clean_image,
        keep_boxes,
        preserve_original_size=True,
        facade_quad=facade_quad,
        validate_rectification=config.validate_facade_rectification,
        min_improvement_deg=config.rectification_min_improvement_deg,
        max_axis_deviation_deg=config.rectification_max_axis_deviation_deg,
    )
    aligned_facade = rectification.aligned_facade
    transform_matrix = rectification.transform_matrix
    src_corners = rectification.source_corners
    rectified_content_mask = rectification.content_mask
    stages["geometry"] = {
        **rectification.quality,
        "building_bbox": [float(bx1), float(by1), float(bx2), float(by2)],
    }
    reconstructed_mask = warp_mask(robust_mask, transform_matrix, aligned_facade.shape)

    print("Stage 7/8 - Facade element segmentation and alignment")
    segmentation = segment_facade_components(
        aligned_facade,
        models["mask_generator"],
        models["predictor"],
        models["dino_model"],
        device,
        (bx1, by1, bx2, by2),
        transform_matrix,
        min_window_detections=config.min_window_detections,
        use_cv_window_fallback=config.use_cv_window_fallback,
        cv_window_min_area_fraction=config.cv_window_min_area_fraction,
        cv_window_max_area_fraction=config.cv_window_max_area_fraction,
        reconstructed_mask=reconstructed_mask,
        preserve_observed_window_geometry=config.preserve_observed_window_geometry,
        infer_windows_in_reconstructed_regions=config.infer_windows_in_reconstructed_regions,
        infer_evidence_supported_windows=config.infer_evidence_supported_windows,
        use_window_grid_regularization=config.use_window_grid_regularization,
        use_uniform_window_grid=config.use_uniform_window_grid,
    )
    trained_segmentation, trained_parser_stage = _segmentation_from_trained_parser(
        aligned_facade,
        config,
        device,
    )
    if trained_segmentation is not None:
        segmentation = _merge_trained_openings_into_segmentation(
            segmentation,
            trained_segmentation,
        )
        trained_parser_stage = {
            **trained_parser_stage,
            "status": "used-for-openings",
            "integration": "base-facade-mask-plus-trained-openings",
        }
    stages["trained_facade_parser"] = trained_parser_stage
    if rectified_content_mask is not None:
        segmentation["facade_mask"] &= rectified_content_mask
        segmentation["window_mask"] &= segmentation["facade_mask"]
        if "raw_window_mask" in segmentation:
            segmentation["raw_window_mask"] &= segmentation["facade_mask"]
        segmentation["door_mask"] &= segmentation["facade_mask"]
        segmentation["balcony_mask"] &= segmentation["facade_mask"]
        if "roof_mask" in segmentation:
            segmentation["roof_mask"] &= segmentation["facade_mask"]
    segmentation = _clean_architectural_exclusions(
        segmentation,
        reconstructed_mask,
        config,
    )
    segmentation = _postprocess_facade_mask(segmentation, aligned_facade)
    stages["segmentation"] = segmentation["quality"]

    # Scaling and floor inference must use the final semantic opening mask,
    # including SAM/CV detections and conservative hidden-window inference,
    # rather than only the smaller set of direct DINO boxes.
    window_boxes_np = _window_boxes_from_mask(
        segmentation["window_mask"],
        segmentation["facade_mask"],
    )
    facade_grid = align_facade_grid(window_boxes_np)
    stages["alignment"] = {
        "floor_bands": len(facade_grid["floors"]),
        "window_columns": len(facade_grid["columns"]),
        "grid": facade_grid,
    }

    shadow_analysis = _disabled_shadow_analysis(segmentation["facade_mask"])
    stages["shadow_analysis"] = {
        "status": "disabled",
        "reason": "Current methodology focus excludes shadow and illumination analysis.",
    }

    print("Stage 8/8 - Usable BIPV area estimation")
    dimensions, validation = estimate_real_world_scale(
        aligned_facade,
        window_boxes_np,
        facade_mask=segmentation["facade_mask"],
        window_mask=segmentation["window_mask"],
        ge_width_m=config.ge_width_m,
        ge_height_m=config.ge_height_m,
        require_google_earth_dimensions=config.require_google_earth_dimensions,
        known_floors=config.known_floors,
        floor_height_m=config.floor_height_m,
    )
    stages["scaling"] = {
        "source": dimensions.get("scale_source", validation.get("source")),
        "method": dimensions.get("scale_method", validation.get("method")),
        "confidence": dimensions.get("scale_confidence", validation.get("confidence")),
        "floor_count_source": dimensions.get("floor_count_source"),
        "floor_count_candidates": dimensions.get("floor_count_candidates"),
        "validation": validation,
    }

    if config.exclude_ground_floor:
        segmentation, dimensions = _exclude_ground_floor_from_mask(segmentation, dimensions)

    obstacle_mask_for_area = reconstructed_mask
    if config.exclude_obstacle_area_from_usable:
        obstacle_mask_for_area &= segmentation["facade_mask"]
    else:
        obstacle_mask_for_area = None

    bipv_surface = segment_bipv_surface(
        segmentation["facade_mask"],
        segmentation["window_mask"],
        segmentation["door_mask"],
        segmentation["balcony_mask"] | segmentation.get("roof_mask", np.zeros_like(segmentation["facade_mask"])),
        shadow_mask=shadow_analysis["shadow_mask"],
        obstacle_mask=obstacle_mask_for_area,
        obstacle_exclusion_dilate_kernel=config.obstacle_exclusion_dilate_kernel,
    )
    usable_results = calculate_usable_area(
        segmentation["facade_mask"],
        segmentation["window_mask"],
        segmentation["door_mask"],
        segmentation["balcony_mask"] | segmentation.get("roof_mask", np.zeros_like(segmentation["facade_mask"])),
        shadow_analysis["shadow_mask"],
        dimensions,
        obstacle_mask=obstacle_mask_for_area,
        obstacle_exclusion_dilate_kernel=config.obstacle_exclusion_dilate_kernel,
    )
    panel_capacity = estimate_panel_capacity(
        usable_results["usable_area_m2"],
        panel_efficiency=config.panel_efficiency,
        panel_area_m2=config.panel_area_m2,
        watts_per_panel=config.watts_per_panel,
    )
    energy_yield = estimate_energy_yield(
        panel_capacity["total_capacity_kw"],
        specific_yield_kwh_per_kwp=config.specific_yield_kwh_per_kwp,
        shading_loss_fraction=shadow_analysis.get("shadow_percentage", 0) / 100,
    )
    bipv_scenarios = {
        "status": "disabled",
        "reason": "Shadow and illumination scenario analysis is out of scope for the current methodology focus.",
    }
    stages["area"] = {
        "facade_area_m2": usable_results["facade_area_m2"],
        "usable_area_m2": usable_results["usable_area_m2"],
        "usable_area_reduced_m2": usable_results["usable_area_reduced_m2"],
        "obstacle_exclusion_area_m2": usable_results["obstacle_exclusion_area_m2"],
        "px_to_m2": usable_results["px_to_m2"],
    }

    export_data = prepare_pvsyst_export(
        config.image_path,
        dimensions,
        usable_results,
        panel_capacity,
        shadow_analysis,
        energy_yield=energy_yield,
        validation=validation,
        bipv_scenarios=bipv_scenarios,
        stage_quality=stages,
    )
    save_pvsyst_export(config.output_path, export_data)
    excel_output_path = excel_path_from_json_path(config.output_path)
    save_pvsyst_excel(excel_output_path, export_data)

    if owns_models:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "image_rgb": image_rgb,
        "clean_image": clean_image,
        "aligned_facade": aligned_facade,
        "source_detection": source_detection,
        "source_annotation": annotate(
            image_rgb,
            source_detection.boxes,
            source_detection.logits,
            source_detection.phrases,
        ),
        "obstacle_mask": robust_mask,
        "obstacle_mask_for_area": obstacle_mask_for_area,
        "src_corners": src_corners,
        "rectification": rectification.quality,
        "segmentation": segmentation,
        "shadow_analysis": shadow_analysis,
        "dimensions": dimensions,
        "facade_grid": facade_grid,
        "bipv_surface": bipv_surface,
        "usable_results": usable_results,
        "panel_capacity": panel_capacity,
        "energy_yield": energy_yield,
        "bipv_scenarios": bipv_scenarios,
        "measurement_quality": segmentation["quality"]["measurement_quality"],
        "export_data": export_data,
        "stages": stages,
        "output_path": config.output_path,
        "excel_output_path": excel_output_path,
    }

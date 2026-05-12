"""End-to-end BIPV facade analysis pipeline."""

from __future__ import annotations

import gc

import cv2
import numpy as np
import torch

from .area import calculate_real_world_dimensions, calculate_usable_area, estimate_panel_capacity
from .config import AnalysisConfig
from .detection import annotate, detect_obstacles_and_architecture
from .export import prepare_pvsyst_export, save_pvsyst_export
from .geometry import (
    building_bbox_from_boxes,
    get_vertical_lines,
    rectify_aspect_preserving,
    rectify_to_original_size,
    robust_vanishing_point,
    validate_google_earth_dimensions,
)
from .image_io import load_image_rgb
from .inpainting import build_robust_mask, remove_obstacles, segment_obstacles_with_sam
from .model_loader import load_models
from .segmentation import segment_facade_components
from .shadows import run_shadow_analysis


def run_bipv_analysis(config: AnalysisConfig | None = None, **kwargs):
    """Run the full BIPV analysis.

    Pass either an ``AnalysisConfig`` or keyword arguments accepted by it.
    This function is intended for Google Colab GPU execution.
    """

    config = config or AnalysisConfig(**kwargs)
    models = load_models(load_stable_diffusion=config.run_stable_diffusion)
    device = models["device"]

    image_rgb = load_image_rgb(config.image_path, max_side=config.max_image_side)
    stages = {}

    print("Stage 1/7 - Obstacle and facade detection")
    source_detection = detect_obstacles_and_architecture(
        image_rgb,
        models["dino_model"],
        device,
        facade_roi_bottom=config.facade_roi_bottom,
    )
    height, width = image_rgb.shape[:2]
    bx1, by1, bx2, by2, keep_boxes = building_bbox_from_boxes(
        source_detection.boxes,
        source_detection.keep_ids,
        height,
        width,
        facade_roi_bottom=config.facade_roi_bottom,
    )
    facade_constraint_mask = np.zeros((height, width), dtype=bool)
    if config.constrain_obstacles_to_facade:
        pad_x = int(width * 0.02)
        pad_y = int(height * 0.02)
        fx1 = max(0, int(bx1) - pad_x)
        fy1 = max(0, int(by1) - pad_y)
        fx2 = min(width - 1, int(bx2) + pad_x)
        fy2 = min(height - 1, int(by2) + pad_y)
        facade_constraint_mask[fy1:fy2, fx1:fx2] = True
    else:
        facade_constraint_mask[:, :] = True
    stages["source_detection"] = {
        "detections": len(source_detection.phrases),
        "architectural": len(source_detection.keep_ids),
        "obstacles": len(source_detection.remove_ids),
        "source_building_bbox": [float(bx1), float(by1), float(bx2), float(by2)],
    }

    print("Stage 2/7 - Obstacle segmentation and removal")
    raw_obstacle_mask = segment_obstacles_with_sam(
        image_rgb,
        source_detection.boxes,
        source_detection.remove_ids,
        models["predictor"],
    )
    raw_obstacle_mask &= facade_constraint_mask
    robust_mask = build_robust_mask(
        raw_obstacle_mask,
        dilate_kernel=config.obstacle_dilate_kernel,
        dilate_iters=config.obstacle_dilate_iters,
        shadow_pad_frac=config.obstacle_shadow_pad_frac,
        max_mask_fraction=config.max_obstacle_mask_fraction,
    )
    robust_mask &= facade_constraint_mask
    clean_image = remove_obstacles(
        image_rgb,
        robust_mask,
        models["lama"],
        sd_pipe=models["sd_pipe"],
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

    print("Stage 3/7 - Facade geometry reconstruction and rectification")
    vertical_lines = get_vertical_lines(clean_image)
    vanishing_point = robust_vanishing_point(vertical_lines)
    if config.preserve_original_size:
        aligned_facade, transform_matrix, src_corners, rectified_content_mask = (
            rectify_to_original_size(clean_image, vanishing_point, keep_boxes)
        )
    else:
        aligned_facade, transform_matrix, src_corners = rectify_aspect_preserving(
            clean_image,
            vanishing_point,
            keep_boxes,
        )
        rectified_content_mask = None
    stages["geometry"] = {
        "input_shape": tuple(clean_image.shape),
        "aligned_shape": tuple(aligned_facade.shape),
        "preserve_original_size": config.preserve_original_size,
        "vertical_lines": len(vertical_lines),
        "building_bbox": [float(bx1), float(by1), float(bx2), float(by2)],
    }

    print("Stage 4/7 - Facade, window, door, and balcony segmentation")
    segmentation = segment_facade_components(
        aligned_facade,
        models["mask_generator"],
        models["predictor"],
        models["dino_model"],
        device,
        (bx1, by1, bx2, by2),
        transform_matrix,
        min_window_detections=config.min_window_detections,
    )
    if rectified_content_mask is not None:
        segmentation["facade_mask"] &= rectified_content_mask
        segmentation["window_mask"] &= segmentation["facade_mask"]
        segmentation["door_mask"] &= segmentation["facade_mask"]
        segmentation["balcony_mask"] &= segmentation["facade_mask"]
    stages["segmentation"] = segmentation["quality"]

    window_boxes_np = np.array(
        [
            box.cpu().numpy()
            for box, phrase in zip(segmentation["boxes"], segmentation["phrases"])
            if "window" in phrase.lower()
        ]
    )
    print("Stage 5/7 - Google Earth scale validation")
    validation = validate_google_earth_dimensions(
        aligned_facade,
        window_boxes_np,
        config.ge_width_m,
        config.ge_height_m,
        config.floor_height_m,
        facade_mask=segmentation["facade_mask"],
    )
    dimensions = calculate_real_world_dimensions(
        aligned_facade,
        window_boxes_np,
        known_floors=config.known_floors,
        reference_height_m=config.floor_height_m,
        validated_width_m=validation["width_m"],
        validated_height_m=validation["height_m"],
        facade_mask=segmentation["facade_mask"],
    )

    print("Stage 6/7 - Shadow detection and usable facade area")
    shadow_analysis = run_shadow_analysis(aligned_facade, segmentation["facade_mask"])
    if transform_matrix is None:
        obstacle_mask_for_area = robust_mask
    else:
        obstacle_mask_for_area = cv2.warpPerspective(
            robust_mask.astype("uint8"),
            transform_matrix,
            (aligned_facade.shape[1], aligned_facade.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)

    if config.exclude_obstacle_area_from_usable:
        obstacle_mask_for_area &= segmentation["facade_mask"]
    else:
        obstacle_mask_for_area = None

    usable_results = calculate_usable_area(
        segmentation["facade_mask"],
        segmentation["window_mask"],
        segmentation["door_mask"],
        segmentation["balcony_mask"],
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
    stages["area"] = {
        "facade_area_m2": usable_results["facade_area_m2"],
        "usable_area_m2": usable_results["usable_area_m2"],
        "usable_area_reduced_m2": usable_results["usable_area_reduced_m2"],
        "obstacle_exclusion_area_m2": usable_results["obstacle_exclusion_area_m2"],
        "px_to_m2": usable_results["px_to_m2"],
    }

    print("Stage 7/7 - PVsyst export")
    export_data = prepare_pvsyst_export(
        config.image_path,
        dimensions,
        usable_results,
        panel_capacity,
        shadow_analysis,
        validation=validation,
    )
    save_pvsyst_export(config.output_path, export_data)

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
        "segmentation": segmentation,
        "shadow_analysis": shadow_analysis,
        "dimensions": dimensions,
        "usable_results": usable_results,
        "panel_capacity": panel_capacity,
        "export_data": export_data,
        "stages": stages,
        "output_path": config.output_path,
    }

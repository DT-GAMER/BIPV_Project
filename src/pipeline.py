"""End-to-end BIPV facade analysis pipeline."""

from __future__ import annotations

import gc

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
from .inpainting import build_robust_mask, remove_obstacles, segment_obstacles_with_sam
from .model_loader import load_models
from .preprocessing import load_and_preprocess_image
from .scaling import estimate_real_world_scale
from .segmentation import segment_facade_components


def _disabled_shadow_analysis(facade_mask):
    """Return a neutral shadow result while shadow analysis is out of scope."""

    shadow_mask = np.zeros_like(facade_mask, dtype=bool)
    return {
        "shadow_area_px": 0,
        "shadow_percentage": 0.0,
        "shadow_mask": shadow_mask,
        "status": "disabled",
    }


def run_bipv_analysis(config: AnalysisConfig | None = None, models=None, **kwargs):
    """Run the full BIPV analysis.

    Pass either an ``AnalysisConfig`` or keyword arguments accepted by it.
    This function is intended for Google Colab GPU execution.
    """

    config = config or AnalysisConfig(**kwargs)
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

    print("Stage 3/8 - Image segmentation")
    raw_obstacle_mask = segment_obstacles_with_sam(
        image_rgb,
        source_detection.boxes,
        source_detection.remove_ids,
        models["predictor"],
    )
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
        "robust_obstacle_pixels": int(robust_mask.sum()),
    }

    print("Stage 5/8 - Obstacle removal / inpainting")
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

    print("Stage 6/8 - Facade rectification")
    # Metric area estimates require rectification to keep the original image
    # canvas and detected building footprint stable.
    rectification = rectify_facade(
        clean_image,
        keep_boxes,
        preserve_original_size=True,
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

    obstacle_mask_for_area = reconstructed_mask
    if config.exclude_obstacle_area_from_usable:
        obstacle_mask_for_area &= segmentation["facade_mask"]
    else:
        obstacle_mask_for_area = None

    bipv_surface = segment_bipv_surface(
        segmentation["facade_mask"],
        segmentation["window_mask"],
        segmentation["door_mask"],
        segmentation["balcony_mask"],
        shadow_mask=shadow_analysis["shadow_mask"],
        obstacle_mask=obstacle_mask_for_area,
        obstacle_exclusion_dilate_kernel=config.obstacle_exclusion_dilate_kernel,
    )
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
        "export_data": export_data,
        "stages": stages,
        "output_path": config.output_path,
        "excel_output_path": excel_output_path,
    }

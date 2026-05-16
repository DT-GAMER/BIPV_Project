"""Export helpers for PVsyst-style project data."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def prepare_pvsyst_export(
    image_path: str,
    dimensions,
    usable_results,
    panel_capacity,
    shadow_analysis,
    validation=None,
):
    return {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "image_path": image_path,
            "analysis_tool": "BIPV Facade Analysis Pipeline",
            "validation": validation or {},
        },
        "building_dimensions": {
            "width_m": dimensions["width_m"],
            "height_m": dimensions["height_m"],
            "total_facade_area_m2": dimensions["total_facade_area_m2"],
            "num_floors": dimensions["num_floors"],
            "scale_source": dimensions.get("scale_source", "measured-or-estimated"),
            "scale_method": dimensions.get("scale_method", "unknown"),
            "scale_confidence": dimensions.get("scale_confidence"),
        },
        "usable_area": {
            "facade_area_m2": usable_results["facade_area_m2"],
            "usable_area_m2": usable_results["usable_area_m2"],
            "usable_area_reduced_m2": usable_results["usable_area_reduced_m2"],
            "usable_percentage": usable_results["usable_percentage"],
            "px_to_m2": usable_results.get("px_to_m2", 0),
            "facade_area_px": usable_results.get("facade_area_px", 0),
            "window_area_m2": usable_results["window_area_m2"],
            "door_area_m2": usable_results["door_area_m2"],
            "balcony_area_m2": usable_results["balcony_area_m2"],
            "shadow_area_m2": usable_results["shadow_area_m2"],
            "obstacle_exclusion_area_m2": usable_results.get("obstacle_exclusion_area_m2", 0),
        },
        "pv_system_estimate": panel_capacity,
        "shadow_analysis": {
            "shadow_area_px": shadow_analysis.get("shadow_area_px", 0),
            "shadow_percentage": shadow_analysis.get("shadow_percentage", 0),
        },
        "pvsyst_inputs": {
            "available_area_m2": usable_results["usable_area_m2"],
            "shadow_reduced_area_m2": usable_results["usable_area_reduced_m2"],
            "estimated_capacity_kw": panel_capacity["total_capacity_kw"],
        },
    }


def save_pvsyst_export(path: str, data) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

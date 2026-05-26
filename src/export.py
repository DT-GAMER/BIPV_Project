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
    energy_yield=None,
    validation=None,
    bipv_scenarios=None,
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
        "energy_yield": energy_yield or {},
        "bipv_scenarios": bipv_scenarios or {},
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


def excel_path_from_json_path(path: str) -> str:
    """Return the matching .xlsx path for a JSON export path."""

    return str(Path(path).with_suffix(".xlsx"))


def _append_mapping_sheet(workbook, title: str, mapping) -> None:
    sheet = workbook.create_sheet(title)
    sheet.append(["field", "value"])
    for key, value in mapping.items():
        sheet.append([key, _excel_cell_value(value)])


def _excel_cell_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value


def _autosize_columns(workbook) -> None:
    for sheet in workbook.worksheets:
        for column_cells in sheet.columns:
            width = max(len(str(cell.value or "")) for cell in column_cells)
            sheet.column_dimensions[column_cells[0].column_letter].width = min(width + 3, 48)


def save_pvsyst_excel(path: str, data) -> None:
    """Save PVsyst-ready data as an Excel workbook."""

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError as exc:
        raise ImportError(
            "Excel export requires openpyxl. Install it with `pip install openpyxl`."
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"
    summary.append(["metric", "value"])
    summary["A1"].font = Font(bold=True)
    summary["B1"].font = Font(bold=True)

    dimensions = data.get("building_dimensions", {})
    usable = data.get("usable_area", {})
    pv_system = data.get("pv_system_estimate", {})
    energy = data.get("energy_yield", {})
    metadata = data.get("metadata", {})
    validation = metadata.get("validation", {})

    summary_rows = {
        "image_path": metadata.get("image_path"),
        "analysis_date": metadata.get("analysis_date"),
        "width_m": dimensions.get("width_m"),
        "height_m": dimensions.get("height_m"),
        "total_facade_area_m2": dimensions.get("total_facade_area_m2"),
        "num_floors": dimensions.get("num_floors"),
        "scale_source": dimensions.get("scale_source"),
        "scale_method": dimensions.get("scale_method"),
        "scale_confidence": dimensions.get("scale_confidence"),
        "usable_area_m2": usable.get("usable_area_m2"),
        "usable_percentage": usable.get("usable_percentage"),
        "window_area_m2": usable.get("window_area_m2"),
        "door_area_m2": usable.get("door_area_m2"),
        "balcony_area_m2": usable.get("balcony_area_m2"),
        "obstacle_exclusion_area_m2": usable.get("obstacle_exclusion_area_m2"),
        "num_panels": pv_system.get("num_panels"),
        "total_capacity_kw": pv_system.get("total_capacity_kw"),
        "panel_area_m2": pv_system.get("panel_area_m2"),
        "watts_per_panel": pv_system.get("watts_per_panel"),
        "annual_kwh": energy.get("annual_kwh"),
        "validation_status": validation.get("status"),
    }
    for key, value in summary_rows.items():
        summary.append([key, _excel_cell_value(value)])

    _append_mapping_sheet(workbook, "PVsyst_Data", data.get("pvsyst_inputs", {}))
    _append_mapping_sheet(workbook, "Building_Dimensions", dimensions)
    _append_mapping_sheet(workbook, "Usable_Area", usable)
    _append_mapping_sheet(workbook, "PV_System", pv_system)
    _append_mapping_sheet(workbook, "Energy_Yield", energy)
    _append_mapping_sheet(workbook, "Validation", validation)

    _autosize_columns(workbook)
    workbook.save(output_path)

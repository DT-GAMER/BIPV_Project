"""Stage 10: simple BIPV capacity and energy estimation."""

from __future__ import annotations


def estimate_panel_capacity(
    usable_area_m2: float,
    panel_efficiency: float = 0.20,
    panel_area_m2: float = 1.7,
    watts_per_panel: int = 350,
):
    """Estimate installable panel count and DC capacity."""

    num_panels = int(usable_area_m2 / panel_area_m2)
    return {
        "num_panels": num_panels,
        "total_capacity_kw": (num_panels * watts_per_panel) / 1000,
        "panel_efficiency": panel_efficiency,
        "panel_area_m2": panel_area_m2,
        "watts_per_panel": watts_per_panel,
    }


def estimate_energy_yield(
    capacity_kw: float,
    specific_yield_kwh_per_kwp: float = 950,
    shading_loss_fraction: float = 0.0,
):
    """Estimate annual energy yield from capacity and simple loss factor."""

    annual_kwh = capacity_kw * specific_yield_kwh_per_kwp * (1 - shading_loss_fraction)
    return {
        "estimated_kwp": capacity_kw,
        "specific_yield_kwh_per_kwp": specific_yield_kwh_per_kwp,
        "shading_loss_fraction": shading_loss_fraction,
        "annual_kwh": annual_kwh,
    }


def _capacity_from_effective_area(
    effective_area_m2: float,
    panel_area_m2: float,
    watts_per_panel: int,
):
    num_panels_equivalent = effective_area_m2 / panel_area_m2 if panel_area_m2 else 0
    capacity_kw = num_panels_equivalent * watts_per_panel / 1000
    return num_panels_equivalent, capacity_kw


def estimate_bipv_scenarios(
    facade_mask,
    window_mask,
    shadow_mask,
    dimensions,
    panel_area_m2: float = 1.7,
    watts_per_panel: int = 350,
    specific_yield_kwh_per_kwp: float = 950,
    window_pv_correction: float = 0.70,
):
    """Estimate BIPV potential under the four paper-style scenarios.

    Scenarios:
    - None: ignores shadows and window-to-wall ratio.
    - Shadow: includes shadow effect only.
    - Window: includes wall/window material difference only.
    - Both: includes both shadows and window-to-wall ratio.

    Window PV is counted with ``window_pv_correction`` relative to wall PV.
    """

    facade_mask = facade_mask.astype(bool)
    window_mask = (window_mask & facade_mask).astype(bool)
    shadow_mask = (shadow_mask & facade_mask).astype(bool)
    wall_mask = facade_mask & ~window_mask

    facade_px = facade_mask.sum()
    px_to_m2 = dimensions["total_facade_area_m2"] / facade_px if facade_px else 0

    wall_area_m2 = wall_mask.sum() * px_to_m2
    window_area_m2 = window_mask.sum() * px_to_m2
    facade_area_m2 = facade_mask.sum() * px_to_m2
    shadow_area_m2 = shadow_mask.sum() * px_to_m2

    illuminated_facade_m2 = (facade_mask & ~shadow_mask).sum() * px_to_m2
    illuminated_wall_m2 = (wall_mask & ~shadow_mask).sum() * px_to_m2
    illuminated_window_m2 = (window_mask & ~shadow_mask).sum() * px_to_m2

    scenario_areas = {
        "none": facade_area_m2,
        "shadow": illuminated_facade_m2,
        "window": wall_area_m2 + window_pv_correction * window_area_m2,
        "both": illuminated_wall_m2 + window_pv_correction * illuminated_window_m2,
    }

    scenarios = {}
    for name, effective_area_m2 in scenario_areas.items():
        equivalent_panels, capacity_kw = _capacity_from_effective_area(
            effective_area_m2,
            panel_area_m2,
            watts_per_panel,
        )
        scenarios[name] = {
            "effective_area_m2": effective_area_m2,
            "equivalent_panels": equivalent_panels,
            "estimated_kwp": capacity_kw,
            "annual_kwh": capacity_kw * specific_yield_kwh_per_kwp,
        }

    return {
        "scenarios": scenarios,
        "inputs": {
            "facade_area_m2": facade_area_m2,
            "wall_area_m2": wall_area_m2,
            "window_area_m2": window_area_m2,
            "shadow_area_m2": shadow_area_m2,
            "window_pv_correction": window_pv_correction,
            "specific_yield_kwh_per_kwp": specific_yield_kwh_per_kwp,
            "panel_area_m2": panel_area_m2,
            "watts_per_panel": watts_per_panel,
        },
        "method": "mask-area-four-scenario-estimate",
    }

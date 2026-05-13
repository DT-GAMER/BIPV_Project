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

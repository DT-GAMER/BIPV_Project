"""Batch execution helpers for multiple facade images."""

from __future__ import annotations

from pathlib import Path

from .config import AnalysisConfig, automatic_config
from .model_loader import load_models
from .pipeline import run_bipv_analysis


def run_batch_analysis(
    image_paths,
    output_dir: str = "/content/drive/MyDrive/BIPV_outputs",
    base_config: AnalysisConfig | None = None,
    max_images: int = 10,
    addresses=None,
    latitudes=None,
    longitudes=None,
    google_maps_api_key: str | None = None,
    geospatial_lookup_radius_m: float | None = None,
    facade_bearing_deg: float | None = None,
    clicked_facade_line_configs=None,
):
    """Run BIPV analysis for multiple images while reusing loaded models.

    Address and coordinate inputs are optional per image. Images with a valid
    address or coordinate pair use geospatial scaling; images without location
    inputs automatically fall back to image-only scale estimation.
    """

    image_paths = list(image_paths)
    addresses = list(addresses) if addresses is not None else None
    latitudes = list(latitudes) if latitudes is not None else None
    longitudes = list(longitudes) if longitudes is not None else None
    clicked_facade_line_configs = (
        list(clicked_facade_line_configs)
        if clicked_facade_line_configs is not None
        else None
    )

    if len(image_paths) == 0:
        raise ValueError("No image paths were provided for batch analysis.")
    if len(image_paths) > max_images:
        raise ValueError(
            f"Batch analysis supports at most {max_images} images per run. "
            f"You provided {len(image_paths)} images. Split them into smaller batches."
        )
    for name, values in (
        ("addresses", addresses),
        ("latitudes", latitudes),
        ("longitudes", longitudes),
        ("clicked_facade_line_configs", clicked_facade_line_configs),
    ):
        if values is not None and len(values) != len(image_paths):
            raise ValueError(
                f"{name} must have the same length as image_paths. "
                f"Got {len(values)} {name} for {len(image_paths)} images."
            )
    if (latitudes is None) != (longitudes is None):
        raise ValueError("latitudes and longitudes must be provided together.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    template = base_config
    models = load_models(
        load_stable_diffusion=template.run_stable_diffusion if template else False
    )

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        image_name = Path(image_path).stem
        output_path = str(output_root / f"{index:02d}_{image_name}_pvsyst_export.json")

        if template is None:
            config = automatic_config(image_path=image_path, output_path=output_path)
        else:
            config = AnalysisConfig(
                **{
                    **template.__dict__,
                    "image_path": image_path,
                    "output_path": output_path,
                }
            )
        config_values = dict(config.__dict__)
        if addresses is not None:
            address = addresses[index - 1]
            config_values.update(
                {
                    "use_geospatial_scaling": bool(address),
                    "address": address,
                    "latitude": None,
                    "longitude": None,
                }
            )
        if latitudes is not None and longitudes is not None:
            latitude = latitudes[index - 1]
            longitude = longitudes[index - 1]
            has_coordinates = latitude is not None and longitude is not None
            config_values.update(
                {
                    "use_geospatial_scaling": has_coordinates,
                    "address": None,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
        if google_maps_api_key is not None:
            config_values["google_maps_api_key"] = google_maps_api_key
        if geospatial_lookup_radius_m is not None:
            config_values["geospatial_lookup_radius_m"] = geospatial_lookup_radius_m
        if facade_bearing_deg is not None:
            config_values["facade_bearing_deg"] = facade_bearing_deg
        if clicked_facade_line_configs is not None:
            line_config = clicked_facade_line_configs[index - 1] or {}
            config_values.update(line_config)
            if line_config:
                config_values["use_geospatial_scaling"] = True
        config = AnalysisConfig(**config_values)

        print(f"\n=== Batch image {index}/{len(image_paths)}: {image_path} ===")
        print("Geospatial scaling:", config.use_geospatial_scaling)
        results.append(run_bipv_analysis(config, models=models))

    return results

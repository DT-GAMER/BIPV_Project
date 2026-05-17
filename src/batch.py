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
):
    """Run BIPV analysis for multiple images while reusing loaded models."""

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

        print(f"\n=== Batch image {index}/{len(image_paths)}: {image_path} ===")
        results.append(run_bipv_analysis(config, models=models))

    return results

"""Model download and loading functions.

Heavy imports live inside functions so the project remains importable locally
without installing all Colab GPU dependencies.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import torch

from .config import CheckpointConfig


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {path.name}...")
        urllib.request.urlretrieve(url, path)


def ensure_checkpoints(config: CheckpointConfig | None = None) -> CheckpointConfig:
    """Download SAM and Grounding DINO assets if missing."""

    config = config or CheckpointConfig()
    download_file(config.sam_url, config.sam_path)
    download_file(config.dino_config_url, config.dino_config_path)
    download_file(config.dino_weights_url, config.dino_weights_path)
    return config


def patch_pillow_for_colab() -> None:
    """Patch older Colab Pillow utility attributes used by some dependencies."""

    import PIL._util

    if not hasattr(PIL._util, "is_directory"):
        PIL._util.is_directory = os.path.isdir
    if not hasattr(PIL._util, "is_path"):
        PIL._util.is_path = lambda f: isinstance(f, (str, bytes, os.PathLike))


def load_models(
    config: CheckpointConfig | None = None,
    device: str | None = None,
    load_stable_diffusion: bool = True,
):
    """Load all models required by the full pipeline."""

    patch_pillow_for_colab()
    config = ensure_checkpoints(config)
    device = device or get_device()

    from groundingdino.util.inference import load_model as load_dino
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    from simple_lama_inpainting import SimpleLama

    sam = sam_model_registry["vit_h"](checkpoint=str(config.sam_path))
    sam.to(device)

    dino_model = load_dino(
        str(config.dino_config_path),
        str(config.dino_weights_path),
        device=device,
    )

    sd_pipe = None
    if load_stable_diffusion:
        from diffusers import StableDiffusionInpaintPipeline

        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        sd_pipe.to(device)

    return {
        "device": device,
        "sam": sam,
        "predictor": SamPredictor(sam),
        "mask_generator": SamAutomaticMaskGenerator(sam),
        "dino_model": dino_model,
        "sd_pipe": sd_pipe,
        "lama": SimpleLama(),
    }

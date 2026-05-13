"""Configuration objects for the BIPV analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
DINO_CONFIG_URL = (
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/"
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
DINO_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)


@dataclass(frozen=True)
class CheckpointConfig:
    """Local paths and download URLs for model assets."""

    root_dir: Path = Path("checkpoints")
    sam_filename: str = "sam_vit_h.pth"
    dino_config_filename: str = "GroundingDINO_SwinT_OGC.py"
    dino_weights_filename: str = "groundingdino_swint_ogc.pth"
    sam_url: str = SAM_CHECKPOINT_URL
    dino_config_url: str = DINO_CONFIG_URL
    dino_weights_url: str = DINO_WEIGHTS_URL

    @property
    def sam_path(self) -> Path:
        return self.root_dir / self.sam_filename

    @property
    def dino_config_path(self) -> Path:
        return self.root_dir / self.dino_config_filename

    @property
    def dino_weights_path(self) -> Path:
        return self.root_dir / self.dino_weights_filename


@dataclass(frozen=True)
class AnalysisConfig:
    """User-facing knobs for one facade analysis run."""

    image_path: str
    output_path: str = "/content/drive/MyDrive/BIPV_images/pvsyst_export.json"
    ge_width_m: float | None = None
    ge_height_m: float | None = None
    require_google_earth_dimensions: bool = False
    known_floors: int | None = None
    floor_height_m: float = 3.0
    panel_efficiency: float = 0.20
    panel_area_m2: float = 1.7
    watts_per_panel: int = 350
    run_stable_diffusion: bool = True
    visualize: bool = True
    preserve_original_size: bool = True
    min_window_detections: int = 3
    facade_roi_bottom: float = 0.90
    max_image_side: int | None = 1280
    constrain_obstacles_to_facade: bool = True
    obstacle_dilate_kernel: int = 9
    obstacle_dilate_iters: int = 1
    obstacle_shadow_pad_frac: float = 0.05
    max_obstacle_mask_fraction: float = 0.22
    exclude_obstacle_area_from_usable: bool = True
    obstacle_exclusion_dilate_kernel: int = 15
    use_cv_window_fallback: bool = True
    cv_window_min_area_fraction: float = 0.00020
    cv_window_max_area_fraction: float = 0.02000

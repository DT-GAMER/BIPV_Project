# BIPV Facade Analysis Pipeline

This project turns a Google Colab notebook experiment into a structured Python
codebase for analysing building facades for BIPV installation potential.

The full pipeline is designed to run on **Google Colab with GPU**. Local
development in VS Code is for editing, linting, documentation, Git, and small
utility tests.

## What It Does

The pipeline converts a raw building photo into an engineering surface for BIPV:

```text
Facade image
  -> object detection
  -> precise segmentation
  -> obstacle removal
  -> perspective transformation
  -> facade alignment
  -> final BIPV surface segmentation
  -> shadow analysis
  -> automatic or measured scaling
  -> energy estimation
  -> export
```

Each stage solves one problem:

1. `src/preprocessing.py` loads, resizes, and normalizes the input image.
2. `src/detection.py` detects facade objects and obstacles with Grounding DINO.
3. `src/segmentation.py` converts detections into masks with SAM and window fallbacks.
4. `src/inpainting.py` removes obstacles with TELEA, LaMa, and optional Stable Diffusion.
5. `src/geometry.py` rectifies camera perspective and aligns the facade plane.
6. `src/alignment.py` structures floors and window columns.
7. `src/bipv_segmentation.py` builds the final usable BIPV mask.
8. `src/shadows.py` estimates shadow coverage.
9. `src/scale_estimation.py` and `src/scaling.py` infer metres automatically, with optional Google Earth validation.
10. `src/energy.py` estimates panel capacity and annual energy.
11. `src/export.py` writes JSON/PVsyst-style outputs.

## Project Structure

```text
BIPV_Project/
  notebooks/
    BIPV_Colab_Run.ipynb
  src/
    config.py
    image_io.py
    model_loader.py
    preprocessing.py
    detection.py
    segmentation.py
    inpainting.py
    geometry.py
    alignment.py
    bipv_segmentation.py
    shadows.py
    scale_estimation.py
    scaling.py
    area.py
    energy.py
    export.py
    pipeline.py
    utils.py
    visualization.py
  requirements-colab.txt
  requirements-dev.txt
  README.md
```

## Recommended Development Flow

Use this project as a GitHub-backed Colab project:

```text
VS Code edit -> commit -> push to GitHub -> Colab pull latest -> run
```

Local machine:

```bash
git add .
git commit -m "Structure BIPV Colab pipeline"
git push
```

Google Colab:

```python
!git clone https://github.com/YOUR_USERNAME/BIPV_Project.git
%cd BIPV_Project
!pip install -r requirements-colab.txt
```

For later updates inside Colab:

```python
%cd /content/BIPV_Project
!git pull
```

## Colab Usage

Open `notebooks/BIPV_Colab_Run.ipynb` in Google Colab and update the image path:

```python
IMAGE_PATH = "/content/drive/MyDrive/BIPV_images/Build5.jpeg"
OUTPUT_PATH = "/content/drive/MyDrive/BIPV_images/pvsyst_export.json"
```

Run automatic upload mode:

```python
from src.config import automatic_config
from src.pipeline import run_bipv_analysis

config = automatic_config(
    image_path=IMAGE_PATH,
    output_path=OUTPUT_PATH,
)

result = run_bipv_analysis(config)
```

Optional research calibration, if measured dimensions are available:

```python
from src.config import AnalysisConfig

config = AnalysisConfig(
    image_path=IMAGE_PATH,
    output_path=OUTPUT_PATH,
    ge_width_m=42.5,
    ge_height_m=18.0,
    require_google_earth_dimensions=True,
)
```

## Local Development

Install only the light development dependencies locally:

```bash
pip install -r requirements-dev.txt
```

You can locally check syntax without downloading or running the heavy models:

```bash
python -m compileall src
```

## Notes

- The full model stack needs Colab GPU runtime.
- Model checkpoints are downloaded into `checkpoints/` and are ignored by Git.
- Input images, generated outputs, and large weights should not be committed.
- Area conversion maps the real facade dimensions onto the detected facade mask,
  not the whole image canvas. This avoids false metre-square values when the
  original-size rectified image includes sky, road, or background.
- With only one uploaded image, absolute metre-square values are estimated from
  detected floors, facade aspect ratio, window evidence, and architectural priors.
  This is the no-manual-input operating mode. Measured Google Earth dimensions
  can still be supplied later as validation labels to report height/width/area error.
- The window stage uses Grounding DINO first, then SAM and CV glass-rectangle
  fallbacks inside the facade mask to reduce missed windows.
- Stable Diffusion inpainting can be disabled in `AnalysisConfig`:

```python
AnalysisConfig(..., run_stable_diffusion=False)
```

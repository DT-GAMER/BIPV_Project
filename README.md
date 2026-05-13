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
  -> Google Earth scaling
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
9. `src/scaling.py` converts pixels to metres using Google Earth dimensions.
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

Open `notebooks/BIPV_Colab_Run.ipynb` in Google Colab and update these values:

```python
IMAGE_PATH = "/content/drive/MyDrive/BIPV_images/Build5.jpeg"
OUTPUT_PATH = "/content/drive/MyDrive/BIPV_images/pvsyst_export.json"
GE_WIDTH_M = None
GE_HEIGHT_M = None
```

If you measure the building in Google Earth, replace `None` with real values:

```python
GE_WIDTH_M = 42.5
GE_HEIGHT_M = 18.0
```

Then run:

```python
from src.config import AnalysisConfig
from src.pipeline import run_bipv_analysis

config = AnalysisConfig(
    image_path=IMAGE_PATH,
    output_path=OUTPUT_PATH,
    ge_width_m=GE_WIDTH_M,
    ge_height_m=GE_HEIGHT_M,
    require_google_earth_dimensions=True,
    preserve_original_size=True,
    min_window_detections=25,
    use_cv_window_fallback=True,
)

result = run_bipv_analysis(config)
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
- For accurate reporting, provide `ge_width_m` and `ge_height_m` from Google
  Earth/Maps and set `require_google_earth_dimensions=True`.
- The window stage uses Grounding DINO first, then SAM and CV glass-rectangle
  fallbacks inside the facade mask to reduce missed windows.
- Stable Diffusion inpainting can be disabled in `AnalysisConfig`:

```python
AnalysisConfig(..., run_stable_diffusion=False)
```

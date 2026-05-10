# BIPV Facade Analysis Pipeline

This project turns a Google Colab notebook experiment into a structured Python
codebase for analysing building facades for BIPV installation potential.

The full pipeline is designed to run on **Google Colab with GPU**. Local
development in VS Code is for editing, linting, documentation, Git, and small
utility tests.

## What It Does

The pipeline:

1. Loads a facade image from Google Drive or another Colab path.
2. Detects architectural facade elements and removable obstacles with Grounding DINO.
3. Segments detected objects with Segment Anything.
4. Removes obstacles using OpenCV TELEA, LaMa, and optional Stable Diffusion inpainting.
5. Rectifies the facade perspective while optionally preserving the original image size.
6. Optionally validates building dimensions using Google Earth measurements.
7. Segments facade, windows, doors, balconies, and shadows.
8. Calculates usable facade area for BIPV in square metres from the facade mask.
9. Estimates panel count and system capacity.
10. Exports a PVsyst-style JSON file.

## Project Structure

```text
BIPV_Project/
  notebooks/
    BIPV_Colab_Run.ipynb
  src/
    config.py
    image_io.py
    model_loader.py
    detection.py
    inpainting.py
    geometry.py
    segmentation.py
    shadows.py
    area.py
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
    preserve_original_size=True,
    min_window_detections=3,
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
- The window stage uses Grounding DINO first, then a conservative SAM fallback
  inside the facade mask if too few windows are detected.
- Stable Diffusion inpainting can be disabled in `AnalysisConfig`:

```python
AnalysisConfig(..., run_stable_diffusion=False)
```

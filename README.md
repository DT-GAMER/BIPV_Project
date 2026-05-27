# An Image-Based Pipeline for BIPV Facade Area Estimation

An image-based pipeline for BIPV facade area estimation, converting street-level
facade images into PVsyst-ready data.

This work develops an image-based pipeline for estimating usable BIPV facade
area from street-level building images. The pipeline integrates object
detection, segmentation, obstacle masking and removal, facade rectification,
facade element segmentation, and real-world area estimation to generate
PVsyst-ready structured outputs in JSON and Excel format.

The full pipeline is designed to run on **Google Colab with GPU**. Local
development in VS Code is for editing, linting, documentation, Git, and small
utility tests.

## What It Does

The pipeline converts a raw building photo into an engineering surface for BIPV:

```text
Facade image
  -> object detection
  -> image segmentation
  -> obstacle masking
  -> obstacle removal
  -> perspective transformation / facade rectification
  -> facade element segmentation
  -> usable BIPV area estimation
  -> JSON and Excel export
```

Each stage solves one problem:

1. `src/preprocessing.py` loads, resizes, and normalizes the input image.
2. `src/detection.py` detects facade objects and obstacles with Grounding DINO.
3. `src/segmentation.py` converts detections into masks with SAM and window fallbacks.
4. `src/inpainting.py` expands obstacle masks for removal.
5. `src/inpainting.py` removes obstacles with TELEA, LaMa, and optional Stable Diffusion.
6. `src/geometry.py` rectifies camera perspective by estimating facade boundaries, vertical structure, and a homography transform.
7. `src/segmentation.py` and `src/alignment.py` segment facade elements and structure floors/window columns.
8. `src/bipv_segmentation.py`, `src/scale_estimation.py`, and `src/area.py` estimate usable BIPV facade area.

Shadow and illumination analysis is currently disabled so development can focus
on the image-based facade parsing and usable-area stages.

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

print(result["output_path"])        # structured JSON
print(result["excel_output_path"])  # PVsyst-ready Excel workbook
```

Batch mode for two or more images:

```python
from src.batch import run_batch_analysis
from src.visualization import show_workflow_grid

image_paths = [
    "/content/drive/MyDrive/BIPV_images/building_1.jpg",
    "/content/drive/MyDrive/BIPV_images/building_2.jpg",
    "/content/drive/MyDrive/BIPV_images/building_3.jpg",
]

results = run_batch_analysis(image_paths)
show_workflow_grid(results, column_titles=["Case 1", "Case 2", "Case 3"])
```

The workflow grid uses the same five rows as the reference figure:

```text
Facade Image
Obstacle Detection
Obstacle Removal
Facade Alignment
Segmentation Result
```

The final outputs are:

```text
Usable BIPV facade area
Facade element segmentation metadata
PVsyst data workbook (.xlsx)
Structured data file (.json)
Workflow/grid image (.jpg)
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
- See `docs/methodology_alignment.md` for how this codebase maps to the
  referenced facade BIPV methodology and what remains to implement.
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
- Stable Diffusion inpainting is enabled by default for stronger obstacle
  reconstruction. It can be disabled in `AnalysisConfig` when GPU memory is
  limited:

```python
AnalysisConfig(..., run_stable_diffusion=False)
```

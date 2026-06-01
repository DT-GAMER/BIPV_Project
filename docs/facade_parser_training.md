# Facade Parser Training Plan

This project currently uses Grounding DINO + SAM for zero-shot facade parsing.
That is useful for prototyping, but a trained facade parser is the correct
long-term path for calculation-grade window/opening masks.

## Target Classes

Use these classes for the first trained model:

```text
0 balcony
1 door
2 facade_wall
3 obstacle
4 roof_edge
5 window_opening
```

For BIPV area, the most important labels are:

```text
facade_wall
window_opening
door
balcony
```

## Annotation Rules

- Label the visible target facade wall only.
- Label every window/opening as tightly as possible.
- Label glass curtain wall panels as `window_opening` unless they are intended
  to be counted as usable BIPV glazing in a later experiment.
- Label doors and entrances separately from windows.
- Label balconies separately.
- Label foreground trees, vehicles, fences, hedges, signs, and poles as
  `obstacle`.
- If an obstacle hides a window, do not invent the hidden window in the label.
  Hidden windows can be handled by grid inference or by multi-view/manual data.

## Recommended Labeling Tool

Use CVAT, Roboflow, or Label Studio. Export labels as YOLO segmentation format:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

All coordinates must be normalized between 0 and 1.

## Dataset Layout

Create this folder layout:

```text
training/datasets/facade_parser/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each image must have a matching `.txt` label file with the same stem:

```text
images/train/building_001.jpg
labels/train/building_001.txt
```

## Dataset Size

Recommended stages:

```text
50-100 images    prototype only
300-500 images   useful project model
1000+ images     research-grade model
```

Include different facade types:

- modern apartment buildings
- older stone/brick buildings
- wide low-rise facades
- tall narrow facades
- trees, cars, fences, signs, and street furniture
- angled and near-front-facing images

## Train

Install training dependencies:

```bash
pip install -r requirements-training.txt
```

Train:

```bash
python scripts/train_facade_parser.py \
  --data training/facade_parser_dataset.yaml \
  --model yolo11s-seg.pt \
  --epochs 100 \
  --imgsz 1024 \
  --batch 4
```

The best weights will be saved under:

```text
training/runs/facade_parser_yolo/weights/best.pt
```

## Evaluate

```bash
python scripts/evaluate_facade_parser.py \
  --weights training/runs/facade_parser_yolo/weights/best.pt \
  --data training/facade_parser_dataset.yaml \
  --split val
```

Track:

- facade mask mAP / IoU
- window/opening mAP / IoU
- door/balcony mAP / IoU
- window recall
- false opening detections
- usable-area error against manually measured reference values

## Predict

```bash
python scripts/predict_facade_parser.py \
  --weights training/runs/facade_parser_yolo/weights/best.pt \
  --source /path/to/images
```

## Integration With The Current Pipeline

After training in Colab, save the best weights to Google Drive:

```text
/content/drive/MyDrive/BIPV_Project/models/facade_parser.pt
```

The trained parser is currently disabled by default while the main workflow
uses Grounding DINO + SAM + CV/window-grid fallbacks. If you explicitly enable
`use_trained_facade_parser=True`, the analysis pipeline checks for trained
weights in this order:

```text
1. AnalysisConfig.trained_facade_parser_path, if you set it manually
2. models/facade_parser.pt inside the project
3. /content/drive/MyDrive/BIPV_Project/models/facade_parser.pt
```

If a trained model is found and enabled, Stage 7 can use it for opening masks.
If the file is missing or the prediction is not usable, the pipeline falls back
to the Grounding DINO + SAM parser.

Manual test integration:

```python
from src.trained_facade_parser import run_trained_facade_parser

result = run_trained_facade_parser(image_rgb, "models/facade_parser.pt")
```

Normal default pipeline usage does not use the trained parser:

```python
from src.config import automatic_config
from src.pipeline import run_bipv_analysis

config = automatic_config(image_path=IMAGE_PATH, output_path=OUTPUT_PATH)
result = run_bipv_analysis(config)
```

To enable the trained parser experiment with a custom weights path:

```python
import dataclasses

config = automatic_config(image_path=IMAGE_PATH, output_path=OUTPUT_PATH)
config = dataclasses.replace(
    config,
    use_trained_facade_parser=True,
    trained_facade_parser_path="/content/drive/MyDrive/BIPV_Project/models/facade_parser.pt",
)
```

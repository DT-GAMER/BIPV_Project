"""Train a facade-specific YOLO segmentation parser.

Example:
    python scripts/train_facade_parser.py \
        --data training/facade_parser_dataset.yaml \
        --model yolo11s-seg.pt \
        --epochs 100 \
        --imgsz 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train facade parser segmentation model.")
    parser.add_argument("--data", default="training/facade_parser_dataset.yaml")
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--project", default="training/runs")
    parser.add_argument("--name", default="facade_parser_yolo")
    parser.add_argument("--device", default=None)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Training requires ultralytics. Install with "
            "`pip install -r requirements-training.txt`."
        ) from exc

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = YOLO(args.model)
    project_path = Path(args.project).resolve()
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_path),
        name=args.name,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        task="segment",
    )


if __name__ == "__main__":
    main()

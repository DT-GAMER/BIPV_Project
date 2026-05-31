"""Evaluate a trained facade parser segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained facade parser.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    parser.add_argument("--data", default="training/facade_parser_dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires ultralytics. Install with "
            "`pip install -r requirements-training.txt`."
        ) from exc

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,
        device=args.device,
        task="segment",
    )
    print(metrics)


if __name__ == "__main__":
    main()

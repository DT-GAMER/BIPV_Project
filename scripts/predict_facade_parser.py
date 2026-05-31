"""Run a trained facade parser on one image or folder."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Predict facade masks with trained parser.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    parser.add_argument("--source", required=True, help="Image path or folder.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--project", default="outputs/facade_parser_predictions")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Prediction requires ultralytics. Install with "
            "`pip install -r requirements-training.txt`."
        ) from exc

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        project=args.project,
        name=args.name,
        device=args.device,
        task="segment",
        save=True,
        save_txt=True,
        save_conf=True,
    )


if __name__ == "__main__":
    main()

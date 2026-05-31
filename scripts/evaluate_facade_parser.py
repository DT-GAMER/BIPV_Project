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


def resolve_weights(path: str) -> Path:
    """Resolve expected or Ultralytics-nested best.pt paths."""

    requested = Path(path)
    if requested.exists():
        return requested

    candidates = [
        Path("runs/segment") / requested,
        Path("training/runs/facade_parser_yolo/weights/best.pt"),
        Path("runs/segment/training/runs/facade_parser_yolo/weights/best.pt"),
    ]
    candidates.extend(Path(".").glob("**/facade_parser_yolo/weights/best.pt"))
    existing = [candidate for candidate in candidates if candidate.exists()]
    if existing:
        return max(existing, key=lambda candidate: candidate.stat().st_mtime)

    raise FileNotFoundError(
        f"Weights not found: {requested}. Expected best.pt under "
        "training/runs/facade_parser_yolo/weights/ or runs/segment/..."
    )


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires ultralytics. Install with "
            "`pip install -r requirements-training.txt`."
        ) from exc

    weights = resolve_weights(args.weights)
    print(f"Using weights: {weights}")

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

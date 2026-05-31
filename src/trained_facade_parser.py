"""Optional trained facade parser integration.

This module is intentionally lazy: it imports Ultralytics only when a trained
model is used. The current production pipeline can continue to run with
Grounding DINO + SAM while a facade-specific model is being trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


CLASS_NAMES = {
    0: "balcony",
    1: "door",
    2: "facade_wall",
    3: "obstacle",
    4: "roof_edge",
    5: "window_opening",
}


@dataclass(frozen=True)
class TrainedFacadeParserResult:
    facade_mask: np.ndarray
    window_mask: np.ndarray
    door_mask: np.ndarray
    balcony_mask: np.ndarray
    obstacle_mask: np.ndarray
    quality: dict


def _resize_mask(mask, shape):
    height, width = shape[:2]
    if mask.shape == (height, width):
        return mask.astype(bool)
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def run_trained_facade_parser(
    image_rgb,
    weights_path: str,
    conf: float = 0.25,
    imgsz: int = 1024,
    device: str | None = None,
) -> TrainedFacadeParserResult:
    """Run trained YOLO segmentation model and return pipeline masks."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Trained facade parser requires ultralytics. Install with "
            "`pip install -r requirements-training.txt`."
        ) from exc

    height, width = image_rgb.shape[:2]
    masks_by_class = {
        "facade_wall": np.zeros((height, width), dtype=bool),
        "window_opening": np.zeros((height, width), dtype=bool),
        "door": np.zeros((height, width), dtype=bool),
        "balcony": np.zeros((height, width), dtype=bool),
        "obstacle": np.zeros((height, width), dtype=bool),
    }
    detections = []

    model = YOLO(weights_path)
    results = model.predict(
        source=image_rgb,
        conf=conf,
        imgsz=imgsz,
        device=device,
        task="segment",
        verbose=False,
    )

    if not results:
        return TrainedFacadeParserResult(
            facade_mask=masks_by_class["facade_wall"],
            window_mask=masks_by_class["window_opening"],
            door_mask=masks_by_class["door"],
            balcony_mask=masks_by_class["balcony"],
            obstacle_mask=masks_by_class["obstacle"],
            quality={"detections": 0, "status": "no-detections"},
        )

    result = results[0]
    if result.masks is None or result.boxes is None:
        return TrainedFacadeParserResult(
            facade_mask=masks_by_class["facade_wall"],
            window_mask=masks_by_class["window_opening"],
            door_mask=masks_by_class["door"],
            balcony_mask=masks_by_class["balcony"],
            obstacle_mask=masks_by_class["obstacle"],
            quality={"detections": 0, "status": "no-masks"},
        )

    mask_data = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for mask, class_id, score in zip(mask_data, classes, confs):
        class_name = CLASS_NAMES.get(int(class_id))
        if class_name is None:
            continue
        mask_bool = _resize_mask(mask > 0.5, image_rgb.shape)
        if class_name in masks_by_class:
            masks_by_class[class_name] |= mask_bool
        detections.append({"class": class_name, "confidence": float(score)})

    facade_mask = masks_by_class["facade_wall"]
    if facade_mask.sum() == 0:
        # If the model only predicts openings, keep the run explicit rather than
        # guessing a facade; the main pipeline can fall back to DINO/SAM.
        status = "no-facade-mask"
    else:
        status = "ok"

    return TrainedFacadeParserResult(
        facade_mask=facade_mask,
        window_mask=masks_by_class["window_opening"] & facade_mask,
        door_mask=masks_by_class["door"] & facade_mask,
        balcony_mask=masks_by_class["balcony"] & facade_mask,
        obstacle_mask=masks_by_class["obstacle"],
        quality={
            "detections": len(detections),
            "status": status,
            "classes": detections,
        },
    )

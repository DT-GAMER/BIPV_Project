"""Grounding DINO detection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .utils import decode_box


ALL_CLASSES = [
    "window",
    "door",
    "wall",
    "balcony",
    "column",
    "roof edge",
    "tree",
    "person",
    "car",
    "vehicle",
    "automobile",
    "bicycle",
    "motorcycle",
    "street light",
    "lamp post",
    "pole",
    "railing",
    "fence",
    "sign",
]

KEEP_KEYWORDS = {"window", "door", "wall", "column", "roof", "balcony"}
REMOVE_KEYWORDS = {
    "person",
    "car",
    "vehicle",
    "automobile",
    "bicycle",
    "motorcycle",
    "tree",
    "bush",
    "shrub",
    "hedge",
    "plant",
    "flower",
    "pole",
    "lamp",
    "street",
    "sign",
    "fence",
    "railing",
    "gate",
    "post",
}

_DINO_TRANSFORM = T.Compose(
    [
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@dataclass
class DetectionResult:
    boxes: torch.Tensor
    logits: torch.Tensor
    phrases: list[str]
    keep_ids: list[int]
    remove_ids: list[int]


def preprocess_for_dino(image_rgb: np.ndarray, device: str) -> torch.Tensor:
    return _DINO_TRANSFORM(Image.fromarray(image_rgb)).to(device)


def detect_obstacles_and_architecture(
    image_rgb: np.ndarray,
    dino_model,
    device: str,
    facade_roi_bottom: float = 0.90,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    extra_classes: list[str] | None = None,
) -> DetectionResult:
    """Detect architectural elements and removable obstacles in the source image."""

    from groundingdino.util.inference import predict as dino_predict

    image_tensor = preprocess_for_dino(image_rgb, device)
    classes = ALL_CLASSES + list(extra_classes or [])
    boxes_all, logits_all, phrases_all = dino_predict(
        model=dino_model,
        image=image_tensor,
        caption=" . ".join(classes),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    facade_boxes, facade_logits, facade_phrases = [], [], []
    for box, logit, phrase in zip(boxes_all, logits_all, phrases_all):
        if box[1].item() <= facade_roi_bottom:
            facade_boxes.append(box)
            facade_logits.append(logit)
            facade_phrases.append(phrase)

    boxes = torch.stack(facade_boxes) if facade_boxes else boxes_all[:0]
    logits = torch.stack(facade_logits) if facade_logits else logits_all[:0]
    phrases = facade_phrases

    keep_ids, remove_ids = [], []
    for index, phrase in enumerate(phrases):
        phrase_lower = phrase.lower()
        if any(keyword in phrase_lower for keyword in REMOVE_KEYWORDS):
            remove_ids.append(index)
        elif any(keyword in phrase_lower for keyword in KEEP_KEYWORDS):
            keep_ids.append(index)

    return DetectionResult(boxes, logits, phrases, keep_ids, remove_ids)


def annotate(image_rgb, boxes, logits, phrases):
    """Draw detections without relying on GroundingDINO/supervision annotation APIs."""

    import cv2

    annotated = image_rgb.copy()
    height, width = image_rgb.shape[:2]
    boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
    logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)

    for box, logit, phrase in zip(boxes_np, logits_np, phrases):
        x1, y1, x2, y2 = decode_box(np.asarray(box), height, width)
        x1 = int(np.clip(round(x1), 0, width - 1))
        y1 = int(np.clip(round(y1), 0, height - 1))
        x2 = int(np.clip(round(x2), 0, width - 1))
        y2 = int(np.clip(round(y2), 0, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        label = f"{phrase} {float(logit):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 2)
        label_y = max(0, y1 - 8)
        (label_w, label_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        cv2.rectangle(
            annotated,
            (x1, max(0, label_y - label_h - baseline)),
            (min(width - 1, x1 + label_w + 6), min(height - 1, label_y + baseline)),
            (34, 197, 94),
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated

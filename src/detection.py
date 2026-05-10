"""Grounding DINO detection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


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
    "pole",
    "lamp",
    "street",
    "sign",
    "fence",
    "railing",
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
) -> DetectionResult:
    """Detect architectural elements and removable obstacles in the source image."""

    from groundingdino.util.inference import predict as dino_predict

    image_tensor = preprocess_for_dino(image_rgb, device)
    boxes_all, logits_all, phrases_all = dino_predict(
        model=dino_model,
        image=image_tensor,
        caption=" . ".join(ALL_CLASSES),
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
    from groundingdino.util.inference import annotate as dino_annotate

    return dino_annotate(image_source=image_rgb, boxes=boxes, logits=logits, phrases=phrases)

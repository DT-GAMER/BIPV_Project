"""Shadow and shading analysis."""

from __future__ import annotations

import cv2
import numpy as np
from skimage import morphology


def detect_shadows_hsv(aligned_facade, threshold_v: float = 0.4, threshold_s: float = 0.3):
    hsv = cv2.cvtColor(aligned_facade, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    return (value < threshold_v) & (saturation < threshold_s)


def detect_shadows_lab(aligned_facade, threshold_l: int = 120):
    lab = cv2.cvtColor(aligned_facade, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    return lightness < threshold_l


def detect_edge_shadows(aligned_facade, kernel_size: int = 15):
    gray = cv2.cvtColor(aligned_facade, cv2.COLOR_RGB2GRAY)
    background = cv2.medianBlur(gray, kernel_size)
    difference = background.astype(np.int16) - gray.astype(np.int16)
    return difference > np.percentile(difference, 85)


def combine_shadow_detections(masks, min_agreement: int = 2):
    votes = np.sum(np.stack(masks, axis=0), axis=0)
    return votes >= min_agreement


def clean_shadow_mask(shadow_mask, min_size: int = 100):
    cleaned = morphology.remove_small_objects(shadow_mask.astype(bool), min_size=min_size)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size)
    return cleaned


def analyze_shadow_patterns(shadow_mask, facade_mask):
    facade_shadow = shadow_mask & facade_mask
    facade_area = facade_mask.sum()
    shadow_area = facade_shadow.sum()
    return {
        "shadow_area_px": int(shadow_area),
        "shadow_percentage": 100 * shadow_area / facade_area if facade_area else 0,
        "shadow_mask": facade_shadow,
    }


def run_shadow_analysis(aligned_facade, facade_mask):
    hsv = detect_shadows_hsv(aligned_facade)
    lab = detect_shadows_lab(aligned_facade)
    edge = detect_edge_shadows(aligned_facade)
    shadow_mask = clean_shadow_mask(combine_shadow_detections([hsv, lab, edge]))
    analysis = analyze_shadow_patterns(shadow_mask, facade_mask)
    analysis["shadow_mask"] = shadow_mask
    return analysis

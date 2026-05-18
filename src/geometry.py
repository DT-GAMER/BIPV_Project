"""Perspective correction and Google Earth dimension validation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line


@dataclass(frozen=True)
class FacadeRectificationResult:
    """Outputs from the high-level facade rectification stage."""

    aligned_facade: np.ndarray
    transform_matrix: np.ndarray
    source_corners: np.ndarray
    content_mask: np.ndarray | None
    quality: dict


def get_vertical_lines(clean_image, min_length: int = 80, max_angle_from_vertical: int = 15):
    height, _ = clean_image.shape[:2]
    roi = clean_image[: int(height * 0.85), :]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    edges = canny(gray, sigma=1.5, low_threshold=30, high_threshold=90)
    lines = probabilistic_hough_line(edges, threshold=60, line_length=min_length, line_gap=8)

    vertical = []
    for (x0, y0), (x1, y1) in lines:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dy and np.degrees(np.arctan2(dx, dy)) < max_angle_from_vertical:
            vertical.append(((x0, y0), (x1, y1)))
    return vertical


def line_intersection_2d(line_1, line_2):
    (x1, y1), (x2, y2) = line_1
    (x3, y3), (x4, y4) = line_2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def robust_vanishing_point(vertical_lines):
    if len(vertical_lines) < 4:
        return None

    points = []
    for i in range(len(vertical_lines)):
        for j in range(i + 1, len(vertical_lines)):
            point = line_intersection_2d(vertical_lines[i], vertical_lines[j])
            if point is not None:
                points.append(point)

    if not points:
        return None
    return np.median(np.array(points), axis=0)


def building_bbox_from_boxes(
    boxes,
    keep_ids,
    height: int,
    width: int,
    facade_roi_bottom: float = 0.90,
):
    keep_boxes = [boxes[index] for index in keep_ids]
    if not keep_boxes:
        return 0, 0, width - 1, int(height * facade_roi_bottom), []

    box_arrays = [box.cpu().numpy() for box in keep_boxes]
    x_min = max(0, min((box[0] - box[2] / 2) * width for box in box_arrays))
    y_min = max(0, min((box[1] - box[3] / 2) * height for box in box_arrays))
    x_max = min(width - 1, max((box[0] + box[2] / 2) * width for box in box_arrays))
    y_max = min(
        int(height * facade_roi_bottom),
        max((box[1] + box[3] / 2) * height for box in box_arrays),
    )
    return x_min, y_min, x_max, y_max, keep_boxes


def _axis_aligned_facade_bbox(clean_image, keep_boxes, pad_frac: float):
    """Return the original detected facade footprint before perspective edits."""
    height, width = clean_image.shape[:2]
    if not keep_boxes:
        margin = int(min(height, width) * 0.02)
        x_min, y_min, x_max, y_max = margin, margin, width - margin, int(height * 0.80)
    else:
        xs1, ys1, xs2, ys2 = [], [], [], []
        for box in keep_boxes:
            cx, cy, box_width, box_height = box.cpu().numpy()
            xs1.append((cx - box_width / 2) * width)
            ys1.append((cy - box_height / 2) * height)
            xs2.append((cx + box_width / 2) * width)
            ys2.append((cy + box_height / 2) * height)
        pad_x, pad_y = width * pad_frac, height * pad_frac
        x_min = max(0, min(xs1) - pad_x)
        y_min = max(0, min(ys1) - pad_y)
        x_max = min(width - 1, max(xs2) + pad_x)
        y_max = min(height - 1, max(ys2) + pad_y)
    return x_min, y_min, x_max, y_max


def _source_quad_from_keep_boxes(clean_image, vanishing_point, keep_boxes, pad_frac: float):
    """Build the facade source quadrilateral from architectural detections."""

    height, width = clean_image.shape[:2]
    x_min, y_min, x_max, y_max = _axis_aligned_facade_bbox(
        clean_image,
        keep_boxes,
        pad_frac,
    )

    src = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)

    if vanishing_point is not None:
        vp_x, vp_y = float(vanishing_point[0]), float(vanishing_point[1])
        vp_above_building = vp_y < y_min - 0.05 * height
        vp_in_image = 0 <= vp_y <= height
        if vp_above_building and not vp_in_image:
            def project(top_x, top_y, bottom_y):
                if abs(top_y - vp_y) < 1e-3:
                    return top_x
                t = (bottom_y - vp_y) / (top_y - vp_y)
                return vp_x + t * (top_x - vp_x)

            bl_x = project(x_min, y_min, y_max)
            br_x = project(x_max, y_min, y_max)
            if x_min - 0.30 * width <= bl_x <= x_max + 0.30 * width and x_min - 0.30 * width <= br_x <= x_max + 0.30 * width:
                src = np.array([[x_min, y_min], [x_max, y_min], [br_x, y_max], [bl_x, y_max]], dtype=np.float32)
    return src


def rectify_aspect_preserving(clean_image, vanishing_point, keep_boxes, pad_frac: float = 0.02):
    """Perspective rectification that avoids stretching width and height differently."""

    height, width = clean_image.shape[:2]
    src = _source_quad_from_keep_boxes(clean_image, vanishing_point, keep_boxes, pad_frac)

    top_w = abs(src[1, 0] - src[0, 0])
    bottom_w = abs(src[2, 0] - src[3, 0])
    left_h = abs(src[3, 1] - src[0, 1])
    right_h = abs(src[2, 1] - src[1, 1])
    natural_w = max(int(max(top_w, bottom_w)), 100)
    natural_h = max(int(max(left_h, right_h)), 100)
    scale = min(width / natural_w, height / natural_h)
    out_w = int(natural_w * scale)
    out_h = int(natural_h * scale)

    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        clean_image,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, matrix, src


def rectify_to_original_size(clean_image, vanishing_point, keep_boxes, pad_frac: float = 0.02):
    """Rectify the facade while preserving the original image canvas size.

    The perspective-adjusted facade quadrilateral is warped back into the
    original detected building footprint on an output canvas with the same
    height and width as the source image. This keeps downstream masks, visual
    outputs, and pixel-to-area calculations tied to the original building size.
    """

    height, width = clean_image.shape[:2]
    src = _source_quad_from_keep_boxes(clean_image, vanishing_point, keep_boxes, pad_frac)
    x_min, y_min, x_max, y_max = _axis_aligned_facade_bbox(
        clean_image,
        keep_boxes,
        pad_frac,
    )
    x_min = max(0, int(np.floor(x_min)))
    x_max = min(width - 1, int(np.ceil(x_max)))
    y_min = max(0, int(np.floor(y_min)))
    y_max = min(height - 1, int(np.ceil(y_max)))

    dst = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        clean_image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    content_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(content_mask, [dst.astype(np.int32)], 1)
    return warped, matrix, src, content_mask.astype(bool)


def _edge_angle_degrees(point_a, point_b):
    dx = float(point_b[0] - point_a[0])
    dy = float(point_b[1] - point_a[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def _source_corner_quality(src_corners, image_shape, vanishing_point, vertical_line_count):
    height, width = image_shape[:2]
    top_angle = _edge_angle_degrees(src_corners[0], src_corners[1])
    bottom_angle = _edge_angle_degrees(src_corners[3], src_corners[2])
    left_angle = _edge_angle_degrees(src_corners[0], src_corners[3])
    right_angle = _edge_angle_degrees(src_corners[1], src_corners[2])
    facade_width_px = float(max(np.linalg.norm(src_corners[1] - src_corners[0]), 1.0))
    facade_height_px = float(max(np.linalg.norm(src_corners[3] - src_corners[0]), 1.0))

    if vanishing_point is None:
        method = "axis-aligned-boundary"
        status = "fallback"
    else:
        method = "vanishing-point-assisted-homography"
        status = "estimated"

    return {
        "rectification_applied": True,
        "boundary_method": "architectural-detection-bbox",
        "perspective_method": method,
        "status": status,
        "vertical_lines": int(vertical_line_count),
        "vanishing_point": (
            None
            if vanishing_point is None
            else [float(vanishing_point[0]), float(vanishing_point[1])]
        ),
        "source_corners": src_corners.astype(float).round(2).tolist(),
        "source_facade_width_px": facade_width_px,
        "source_facade_height_px": facade_height_px,
        "source_facade_fraction": float(
            (facade_width_px * facade_height_px) / max(width * height, 1)
        ),
        "pre_rectification_angles_deg": {
            "top": top_angle,
            "bottom": bottom_angle,
            "left": left_angle,
            "right": right_angle,
        },
    }


def rectify_facade(
    clean_image,
    keep_boxes,
    preserve_original_size: bool = True,
    pad_frac: float = 0.02,
    min_line_length: int = 80,
) -> FacadeRectificationResult:
    """Run the full high-level facade rectification stage.

    This converts an angled facade photo into a more front-facing facade frame:
    detect vertical structure, estimate a vanishing point when possible, build
    the facade source boundary, then apply a homography/perspective transform.
    """

    vertical_lines = get_vertical_lines(clean_image, min_length=min_line_length)
    vanishing_point = robust_vanishing_point(vertical_lines)

    if preserve_original_size:
        aligned_facade, transform_matrix, src_corners, content_mask = rectify_to_original_size(
            clean_image,
            vanishing_point,
            keep_boxes,
            pad_frac=pad_frac,
        )
        output_mode = "preserve-original-size"
    else:
        aligned_facade, transform_matrix, src_corners = rectify_aspect_preserving(
            clean_image,
            vanishing_point,
            keep_boxes,
            pad_frac=pad_frac,
        )
        content_mask = None
        output_mode = "aspect-preserving-crop"

    quality = _source_corner_quality(
        src_corners,
        clean_image.shape,
        vanishing_point,
        len(vertical_lines),
    )
    quality.update(
        {
            "input_shape": tuple(clean_image.shape),
            "aligned_shape": tuple(aligned_facade.shape),
            "output_mode": output_mode,
            "preserve_original_size": preserve_original_size,
            "preserve_building_footprint": preserve_original_size,
            "transform_matrix": transform_matrix.astype(float).round(6).tolist(),
        }
    )

    return FacadeRectificationResult(
        aligned_facade=aligned_facade,
        transform_matrix=transform_matrix,
        source_corners=src_corners,
        content_mask=content_mask,
        quality=quality,
    )


def validate_google_earth_dimensions(
    aligned_facade,
    window_boxes_np,
    ge_width_m: float | None,
    ge_height_m: float | None,
    floor_height_m: float = 3.0,
    facade_mask=None,
    require_google_earth_dimensions: bool = False,
):
    """Return estimated and optionally Google-Earth-validated dimensions."""

    if require_google_earth_dimensions and (ge_width_m is None or ge_height_m is None):
        raise ValueError(
            "Google Earth dimensions are required. Set ge_width_m and ge_height_m "
            "from Google Earth/Maps before running area calculations."
        )

    if len(window_boxes_np) > 0:
        centers_y = np.sort(window_boxes_np[:, 1])
        floors = int(np.clip(int(np.sum(np.diff(centers_y) > 0.06)) + 1, 2, 20))
    else:
        floors = 5

    if facade_mask is None:
        height_px, width_px = aligned_facade.shape[:2]
    else:
        ys, xs = np.where(facade_mask)
        if len(xs) == 0:
            height_px, width_px = aligned_facade.shape[:2]
        else:
            height_px = ys.max() - ys.min() + 1
            width_px = xs.max() - xs.min() + 1
    estimated_height_m = floors * floor_height_m
    pixels_per_meter = height_px / estimated_height_m
    estimated_width_m = width_px / pixels_per_meter

    if ge_width_m is None or ge_height_m is None:
        return {
            "status": "unvalidated",
            "source": "floor-count-estimate",
            "num_floors": floors,
            "height_m": estimated_height_m,
            "width_m": estimated_width_m,
            "pixels_per_meter": pixels_per_meter,
        }

    height_error = abs(estimated_height_m - ge_height_m) / ge_height_m * 100
    width_error = abs(estimated_width_m - ge_width_m) / ge_width_m * 100
    max_error = max(height_error, width_error)
    status = "excellent" if max_error < 5 else "acceptable" if max_error < 10 else "failed"
    return {
        "status": status,
        "source": "google-earth",
        "num_floors": floors,
        "height_m": ge_height_m,
        "width_m": ge_width_m,
        "pixels_per_meter": height_px / ge_height_m,
        "height_error_percent": height_error,
        "width_error_percent": width_error,
    }

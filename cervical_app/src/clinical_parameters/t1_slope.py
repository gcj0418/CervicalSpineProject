"""
T1 Slope calculation.

Input data format
-----------------
pts_front : (28, 2) numpy array, ordered C2 -> T1.
    See cobb_angle.py for the full corner-order and index-mapping specification.

Parameters calculated
---------------------
- T1 slope (°): angle between T1 upper endplate and the horizontal reference line.
  T1 corresponds to vertebra index 6 (last vertebra, pts[24:28]).

  Positive = upper endplate tilts downward to the right (caudal direction).
  Negative = upper endplate tilts upward to the right.

Note on data source
-------------------
This module receives already-normalised keypoints.  Model-output parsing
is handled externally (inference.py / gui.py).
"""
import numpy as np
from .cobb_angle import extract_vertebra_boxes


def compute_t1_slope(pts: np.ndarray) -> float:
    """Compute T1 slope (°).

    Args:
        pts: (28, 2) ordered front points (C2..T1).

    Returns:
        T1 slope in degrees (signed float).
    """
    boxes = extract_vertebra_boxes(pts)

    # T1 = last vertebra (index 6)
    t1 = boxes[6]
    tl, tr, bl, br = t1

    # Upper endplate angle: tr - tl
    angle = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    return float(angle)


def draw_t1_slope(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Draw T1 slope visualization on the image.

    Draws:
      - Magenta box around T1 vertebra.
      - Extended line parallel to T1 upper endplate.
      - Angle label near the line.
    """
    import cv2
    vis = image.copy()
    boxes = extract_vertebra_boxes(pts)

    t1 = boxes[6]
    tl, tr, bl, br = t1
    quad = t1.astype(int)

    color = (255, 0, 255)
    for j in range(4):
        cv2.circle(vis, tuple(quad[j]), 5, color, -1, lineType=cv2.LINE_AA)
    for a, b in [(0, 1), (1, 3), (3, 2), (2, 0)]:
        cv2.line(vis, tuple(quad[a]), tuple(quad[b]), color, 2, lineType=cv2.LINE_AA)

    cx = (tl[0] + tr[0]) / 2.0
    cy = (tl[1] + tr[1]) / 2.0
    angle = compute_t1_slope(pts)
    rad = np.radians(angle)
    length = max(vis.shape[:2]) * 0.4
    dx = np.cos(rad) * length
    dy = np.sin(rad) * length
    p1 = (int(cx - dx), int(cy - dy))
    p2 = (int(cx + dx), int(cy + dy))
    cv2.line(vis, p1, p2, color, 2, lineType=cv2.LINE_AA)

    cv2.putText(vis, f"T1={angle:.1f}°", (int(cx) + 10, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)
    return vis

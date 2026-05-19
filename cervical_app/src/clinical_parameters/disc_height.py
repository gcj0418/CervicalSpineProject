"""
Intervertebral disc height calculation.

Input data format
-----------------
pts_front : (28, 2) numpy array, ordered C2 -> T1.
    See cobb_angle.py for the full corner-order and index-mapping specification.

Parameters calculated
---------------------
- Disc heights (mm) for 6 segments: C2/3, C3/4, C4/5, C5/6, C6/7, C7/T1.
  For each segment, the height is the mid-sagittal Euclidean distance between
  the lower endplate midpoint of the upper vertebra and the upper endplate
  midpoint of the lower vertebra.

Required external data
----------------------
pixel_spacing : float
    mm per pixel, already converted to a scalar by the caller.

Note on data source
-------------------
This module receives already-normalised keypoints.  Model-output parsing
is handled externally (inference.py / gui.py).
"""
import numpy as np
from .cobb_angle import extract_vertebra_boxes


def compute_disc_heights(pts: np.ndarray, pixel_spacing: float = 1.0) -> dict:
    """Compute intervertebral disc heights for all segments (mm).

    Args:
        pts: (28, 2) ordered front points (C2..T1).
        pixel_spacing: scalar mm/pixel.

    Returns:
        dict mapping segment name -> disc height in mm.
    """
    boxes = extract_vertebra_boxes(pts)

    segments = ['C2/3', 'C3/4', 'C4/5', 'C5/6', 'C6/7', 'C7/T1']
    heights = {}

    for i in range(6):
        # Upper vertebra: lower endplate midpoint = (bl + br) / 2
        upper_bl, upper_br = boxes[i][2], boxes[i][3]
        mid_lower = (upper_bl + upper_br) / 2.0

        # Lower vertebra: upper endplate midpoint = (tl + tr) / 2
        lower_tl, lower_tr = boxes[i + 1][0], boxes[i + 1][1]
        mid_upper = (lower_tl + lower_tr) / 2.0

        dist = np.linalg.norm(mid_lower - mid_upper)
        heights[segments[i]] = float(dist * pixel_spacing)

    # Mean disc height across all segments
    if heights:
        heights['mean'] = float(np.mean(list(heights.values())))

    return heights


def draw_disc_heights(image: np.ndarray, pts: np.ndarray, pixel_spacing: float = 1.0) -> np.ndarray:
    """Draw intervertebral disc heights on the image.

    Draws a coloured line connecting the lower endplate midpoint of the upper
    vertebra to the upper endplate midpoint of the lower vertebra, with a
    label showing the segment name and height in mm.
    """
    import cv2
    vis = image.copy()
    boxes = extract_vertebra_boxes(pts)

    heights = compute_disc_heights(pts, pixel_spacing)
    segments = ['C2/3', 'C3/4', 'C4/5', 'C5/6', 'C6/7', 'C7/T1']
    colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 128, 255), (128, 0, 255)]

    for i in range(6):
        upper_bl, upper_br = boxes[i][2], boxes[i][3]
        mid_lower = (upper_bl + upper_br) / 2.0

        lower_tl, lower_tr = boxes[i + 1][0], boxes[i + 1][1]
        mid_upper = (lower_tl + lower_tr) / 2.0

        p1 = tuple(mid_lower.astype(int))
        p2 = tuple(mid_upper.astype(int))
        color = colors[i % len(colors)]

        cv2.line(vis, p1, p2, color, 2, lineType=cv2.LINE_AA)

        # Mark the two midpoints with small dots
        cv2.circle(vis, p1, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, p2, 3, color, -1, lineType=cv2.LINE_AA)

        mid = ((mid_lower + mid_upper) / 2).astype(int)
        label = f"{segments[i]}:{heights[segments[i]]:.1f}mm"
        cv2.putText(vis, label, (mid[0] + 5, mid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

    return vis

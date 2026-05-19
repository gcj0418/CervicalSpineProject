"""
C2-7 SVA (Sagittal Vertical Axis) calculation.

Input data format
-----------------
pts_front : (28, 2) numpy array, ordered C2 -> T1.
    See cobb_angle.py for the full corner-order and index-mapping specification.

Parameters calculated
---------------------
- C2-7 SVA (mm): horizontal distance from the C2 vertebral body centre
  plumb line to the posterior-superior corner of C7.

  In lateral view:
    - anterior (front) is on the left  (smaller x)
    - posterior (back) is on the right (larger x)

  Positive SVA  -> C2 centre is anterior to C7 posterior-superior corner
                   (forward imbalance).
  Negative SVA  -> C2 centre is posterior to C7 posterior-superior corner
                   (backward imbalance).

Required external data
----------------------
pixel_spacing : float
    mm per pixel, extracted from the NIfTI header or image metadata.
    This module assumes spacing is already converted to a single scalar
    (e.g. average in-plane spacing) by the caller.

Note on data source
-------------------
This module receives already-normalised keypoints.  Model-output parsing
is handled externally (inference.py / gui.py).
"""
import numpy as np
from .cobb_angle import extract_vertebra_boxes


def compute_c2c7_sva(pts: np.ndarray, pixel_spacing: float = 1.0) -> float:
    """Compute C2-7 SVA (mm).

    Args:
        pts: (28, 2) ordered front points (C2..T1).
        pixel_spacing: scalar mm/pixel.

    Returns:
        Signed SVA in millimetres.
    """
    boxes = extract_vertebra_boxes(pts)

    # C2 centre = mean of 4 front corners
    c2_center = boxes[0].mean(axis=0)

    # C7 posterior-superior corner = tr (top-right, back-superior)
    c7_ps = boxes[5][1]

    # Horizontal distance: x_C7_ps - x_C2_center
    dx = c7_ps[0] - c2_center[0]
    sva = dx * pixel_spacing
    return float(sva)


def draw_sva(image: np.ndarray, pts: np.ndarray, pixel_spacing: float = 1.0) -> np.ndarray:
    """Draw C2-7 SVA visualization on the image.

    Draws:
      - Red circle at C2 centre.
      - Blue circle at C7 posterior-superior corner.
      - Yellow plumb line from C2 centre downward.
      - Yellow horizontal connector from C7 PS to the plumb line.
      - SVA value label near the connector.
    """
    import cv2
    vis = image.copy()
    boxes = extract_vertebra_boxes(pts)

    c2_center = boxes[0].mean(axis=0)
    c7_ps = boxes[5][1]

    c2_int = tuple(c2_center.astype(int))
    c7_int = tuple(c7_ps.astype(int))

    cv2.circle(vis, c2_int, 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(vis, c7_int, 6, (255, 0, 0), -1, lineType=cv2.LINE_AA)

    h = vis.shape[0]
    cv2.line(vis, c2_int, (c2_int[0], h), (0, 255, 255), 2, lineType=cv2.LINE_AA)

    plumb_intersection = (c2_int[0], c7_int[1])
    cv2.line(vis, c7_int, plumb_intersection, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    sva = compute_c2c7_sva(pts, pixel_spacing)
    mid_x = int((c2_center[0] + c7_ps[0]) / 2)
    mid_y = int(c7_ps[1]) - 10
    cv2.putText(vis, f"SVA={sva:+.1f}mm", (mid_x - 40, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    return vis

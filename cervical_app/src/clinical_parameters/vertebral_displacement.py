"""
Vertebral displacement (listhesis) calculation.

Input data format
-----------------
pts_front : (28, 2) numpy array, ordered C2 -> T1.
    See cobb_angle.py for the full corner-order and index-mapping specification.

Parameters calculated
---------------------
- Displacement (mm) for 6 segments: C2/3, C3/4, C4/5, C5/6, C6/7, C7/T1.
  Relative horizontal displacement between adjacent vertebrae.

  In lateral view:
    - anterior (front) is on the left  (smaller x)
    - posterior (back) is on the right (larger x)

  Positive value  -> upper vertebra slipped anteriorly (forward, toward left).
  Negative value  -> upper vertebra slipped posteriorly (backward, toward right).

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


def compute_vertebral_displacement(pts: np.ndarray, pixel_spacing: float = 1.0) -> dict:
    """Compute vertebral displacement between adjacent levels (mm).

    Args:
        pts: (28, 2) ordered front points (C2..T1).
        pixel_spacing: scalar mm/pixel.

    Returns:
        dict mapping segment name -> displacement in mm (signed).
    """
    boxes = extract_vertebra_boxes(pts)

    segments = ['C2/3', 'C3/4', 'C4/5', 'C5/6', 'C6/7', 'C7/T1']
    displacements = {}

    for i in range(6):
        upper_br = boxes[i][3]      # posterior-inferior corner of upper vertebra
        lower_tr = boxes[i + 1][1]  # posterior-superior corner of lower vertebra

        dx = upper_br[0] - lower_tr[0]
        displacements[segments[i]] = float(dx * pixel_spacing)

    return displacements


def draw_displacements(image: np.ndarray, pts: np.ndarray, pixel_spacing: float = 1.0) -> np.ndarray:
    """Draw vertebral displacements on the image.

    Draws horizontal reference lines and markers at vertebral centres for each segment.
    """
    import cv2
    vis = image.copy()
    boxes = extract_vertebra_boxes(pts)

    displacements = compute_vertebral_displacement(pts, pixel_spacing)
    segments = ['C2/3', 'C3/4', 'C4/5', 'C5/6', 'C6/7', 'C7/T1']
    color = (0, 165, 255)

    for i in range(6):
        upper_br = boxes[i][3]
        lower_tr = boxes[i + 1][1]

        y_ref = int((upper_br[1] + lower_tr[1]) / 2)
        x1, x2 = int(upper_br[0]), int(lower_tr[0])

        # Horizontal reference line connecting posterior corners
        cv2.line(vis, (x1, y_ref), (x2, y_ref), color, 1, lineType=cv2.LINE_AA)

        # Mark posterior corners with vertical ticks
        cv2.line(vis, (x1, int(upper_br[1]) - 5),
                 (x1, int(upper_br[1]) + 5), (0, 0, 255), 2)
        cv2.line(vis, (x2, int(lower_tr[1]) - 5),
                 (x2, int(lower_tr[1]) + 5), (255, 0, 0), 2)

        # Arrow indicating displacement direction
        val = displacements[segments[i]]
        mid_x = int((x1 + x2) / 2)
        arrow_y = y_ref - 8
        if val > 0:  # anterior listhesis (upper slipped forward / left)
            cv2.arrowedLine(vis, (mid_x + 10, arrow_y), (mid_x - 10, arrow_y),
                            color, 1, tipLength=0.4, line_type=cv2.LINE_AA)
        elif val < 0:  # posterior listhesis (upper slipped backward / right)
            cv2.arrowedLine(vis, (mid_x - 10, arrow_y), (mid_x + 10, arrow_y),
                            color, 1, tipLength=0.4, line_type=cv2.LINE_AA)

        label = f"{segments[i]}:{val:+.1f}mm"
        cv2.putText(vis, label, (mid_x - 30, y_ref - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

    return vis

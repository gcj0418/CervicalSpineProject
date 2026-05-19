"""
Facet joint angle calculation.

Input data format
-----------------
vld_pts : (N*4, 2) numpy array — RAW VLD (SpineNet) output.
    N is the number of vertebrae detected by VLD (N >= 7).
    The points are grouped by 4 per vertebra and sorted top-to-bottom
    by the VLD decoder (rearrange_pts in vld_inference.py).

    Each vertebra has 4 corners in order [tl, tr, bl, br]:
        tl = top-left      (front-superior / anterior-superior)
        tr = top-right     (back-superior  / posterior-superior, near upper facet)
        bl = bottom-left   (front-inferior / anterior-inferior)
        br = bottom-right  (back-inferior  / posterior-inferior, near lower facet)

    Why VLD raw output instead of fused 28 pts?
        - VLD outputs high-precision bounding boxes directly.
        - The Hungarian matching in inference.py is designed to pick the
          best 7 boxes for Cobb-angle calculation (where only 7 vertebrae
          matter).  Facet joints, however, can be measured from ALL
          detected posterior edges without being constrained to the
          7-vertebra subset.
        - Therefore facet-joint angles use the VLD raw landmarks,
          while Cobb/SVA/T1/disc-height/displacement use the fused 28 pts.

Parameters calculated
---------------------
- Facet joint angles (°) for 6 segments: C2/3, C3/4, C4/5, C5/6, C6/7, C7/T1.
  For each segment the angle is computed between the posterior edge
  directions of two adjacent vertebrae:
      Upper vertebra: lower posterior edge  (tr -> br)
      Lower vertebra: upper posterior edge  (tr -> br)

Note on data source
-------------------
This module receives the RAW VLD output (N*4 landmarks).  HRNet inference,
Hungarian matching, and left/right orientation unification are handled
externally (inference.py / gui.py).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _reshape_vld_boxes(vld_pts: np.ndarray):
    """Reshape (N*4, 2) flat array into (N, 4, 2) boxes."""
    n = len(vld_pts) // 4
    return vld_pts.reshape(n, 4, 2)


def compute_facet_joint_angles(vld_pts: np.ndarray, pixel_spacing: float = 1.0) -> dict:
    """Compute facet joint angles from VLD raw output (°).

    Args:
        vld_pts: (N*4, 2) VLD raw landmarks, grouped by 4, sorted top-to-bottom.
                 Each vertebra: [tl, tr, bl, br].
        pixel_spacing: scalar mm/pixel (reserved for future physical scaling).

    Returns:
        dict mapping segment name -> facet joint angle in degrees.
    """
    boxes = _reshape_vld_boxes(vld_pts)

    # Take first 7 vertebrae (C2-T1)
    if boxes.shape[0] < 7:
        raise ValueError(f"VLD detected only {boxes.shape[0]} vertebrae, need at least 7")
    boxes = boxes[:7]

    segment_names = ['C2/3', 'C3/4', 'C4/5', 'C5/6', 'C6/7', 'C7/T1']
    angles = {}

    for i in range(6):
        # Upper vertebra: lower posterior edge direction (tr -> br)
        upper_tr, upper_br = boxes[i][1], boxes[i][3]
        v_upper = upper_br - upper_tr

        # Lower vertebra: upper posterior edge direction (tr -> br)
        lower_tr, lower_br = boxes[i + 1][1], boxes[i + 1][3]
        v_lower = lower_br - lower_tr

        # Angle between posterior edge directions
        dot = np.dot(v_upper, v_lower)
        norm_u = np.linalg.norm(v_upper)
        norm_l = np.linalg.norm(v_lower)
        cos_theta = dot / (norm_u * norm_l + 1e-8)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))

        angles[segment_names[i]] = float(angle)

    return angles


def draw_facet_angles(image: np.ndarray, vld_pts: np.ndarray) -> np.ndarray:
    """Draw facet joint angles on the image.

    Draws posterior edge lines (tr -> br) for each vertebra and labels
    each vertebra with its facet-joint name (C2J … T1J).  Segment angles
    are shown between adjacent posterior edges.
    """
    import cv2
    vis = image.copy()
    boxes = _reshape_vld_boxes(vld_pts)

    if boxes.shape[0] < 7:
        return vis
    boxes = boxes[:7]

    angles = compute_facet_joint_angles(vld_pts)
    vertebra_names = ['C2J', 'C3J', 'C4J', 'C5J', 'C6J', 'C7J', 'T1J']
    segment_names = ['C2/3', 'C3/4', 'C4/5', 'C5/6', 'C6/7', 'C7/T1']
    colors = [(128, 128, 0), (0, 128, 128), (128, 0, 128),
              (0, 128, 0), (128, 0, 0), (0, 0, 128),
              (255, 128, 0)]

    # --- Draw all 4 corners and bounding box for each vertebra ---
    corner_labels = ['tl', 'tr', 'bl', 'br']
    for i in range(7):
        tl, tr, bl, br = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        color = colors[i % len(colors)]

        # Draw bounding box polygon (tl-tr-br-bl)
        pts = np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=color,
                      thickness=1, lineType=cv2.LINE_AA)

        # Draw all 4 corner points with labels
        for pt, label in zip([tl, tr, bl, br], corner_labels):
            pt_int = tuple(pt.astype(int))
            cv2.circle(vis, pt_int, 2, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(vis, f"{label}{i}",
                        (pt_int[0] + 4, pt_int[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1,
                        lineType=cv2.LINE_AA)

        # Vertebra name label at posterior-edge (tr-br) midpoint, shifted right
        mid = ((tr + br) / 2).astype(int)
        cv2.putText(vis, vertebra_names[i], (mid[0] + 5, mid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

    # --- Draw angle labels between adjacent posterior edges ---
    # Collect labels to draw via PIL (supports Unicode degree symbol)
    pil_labels = []
    for i in range(6):
        upper_tr, upper_br = boxes[i][1], boxes[i][3]
        lower_tr, lower_br = boxes[i + 1][1], boxes[i + 1][3]
        color = colors[i % len(colors)]

        # Place angle label halfway between the two posterior-edge midpoints
        mid_upper = (upper_tr + upper_br) / 2
        mid_lower = (lower_tr + lower_br) / 2
        label_pos = ((mid_upper + mid_lower) / 2).astype(int)

        label = f"{segment_names[i]}:{angles[segment_names[i]]:.1f}°"
        pil_labels.append((label, (int(label_pos[0] + 5), int(label_pos[1])), color))

    # Render Unicode labels with PIL
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(vis_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for text, pos, color in pil_labels:
        # PIL uses RGB, OpenCV uses BGR
        rgb_color = (color[2], color[1], color[0])
        draw.text(pos, text, font=font, fill=rgb_color)
    vis = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return vis

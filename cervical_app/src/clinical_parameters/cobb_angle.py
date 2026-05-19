"""
Cobb angle calculation for cervical spine.

Input data format
-----------------
pts_front : (28, 2) numpy array, ordered C2 -> T1.
    Each vertebra has 4 corners in order [tl, tr, bl, br]:
        tl = top-left     (front-superior / anterior-superior)
        tr = top-right    (back-superior  / posterior-superior)
        bl = bottom-left  (front-inferior / anterior-inferior)
        br = bottom-right (back-inferior  / posterior-inferior)
    Index mapping (vertex index i -> pts[i*4 : (i+1)*4]):
        C2: i=0  -> pts[0:4]
        C3: i=1  -> pts[4:8]
        C4: i=2  -> pts[8:12]
        C5: i=3  -> pts[12:16]
        C6: i=4  -> pts[16:20]
        C7: i=5  -> pts[20:24]
        T1: i=6  -> pts[24:28]

Parameters calculated
---------------------
- C2-7 Cobb angle (°): signed angle between C2 lower endplate and C7 lower endplate.
  Positive = lordosis (anterior convexity), Negative = kyphotic / reversed.
- Max Cobb angle (°): maximum inter-vertebra angle among C2-C7.
- Diagnosis string: categorical diagnosis based on C2-7 lordosis value.

Visualization
-------------
draw_cobb(image, pts_front): draws vertebra boxes (C2-C7) and perpendicular
reference lines for the C2/C7 lower endplates.

Note on data source
-------------------
This module receives already-normalised keypoints.  How the raw HRNet/VLD
outputs are converted into this standard 28-point array is handled externally
(in inference.py / gui.py).
"""
import numpy as np
import cv2
import os


def normalize_corners(pts: np.ndarray):
    """Normalize each vertebra's 4 corners to tl,tr,bl,br order.
    Uses adjacent spine segment (e.g. C2->C3) as local cranial-caudal axis,
    rotates 90 deg CCW for endplate reference, then assigns 4 corners
    to quadrants (superior/inferior x left/right).
    Does NOT reorder vertebrae (assumes C2..T1 order is correct).
    """
    n = pts.shape[0]
    assert n % 4 == 0, f"Expected N*4 points, got {n}"
    n_v = n // 4
    boxes = [pts[i*4:(i+1)*4].astype(np.float32) for i in range(n_v)]
    centers = [b.mean(axis=0) for b in boxes]

    result = []
    for i in range(n_v):
        box = boxes[i]
        # Local cranial-caudal axis: point to next vertebra (caudal)
        if i < n_v - 1:
            axis_vec = centers[i+1] - centers[i]
        else:
            axis_vec = centers[i] - centers[i-1]

        # Ensure axis points roughly downward (positive y)
        if axis_vec[1] < 0:
            axis_vec = -axis_vec

        center = box.mean(axis=0)
        centered = box - center

        # Perpendicular axis (endplate direction): rotate 90 deg CCW
        perp = np.array([-axis_vec[1], axis_vec[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-6)
        axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-6)

        proj_axis = centered @ axis   # positive = caudal, negative = cranial
        proj_perp = centered @ perp   # positive = left, negative = right

        axis_sort = np.argsort(proj_axis)
        superior_idx = axis_sort[:2]
        inferior_idx = axis_sort[2:]

        # Use raw x-coordinate for left/right split (robust when axis_vec is tilted)
        sup_x = centered[superior_idx, 0]
        inf_x = centered[inferior_idx, 0]
        sup_sort = np.argsort(sup_x)
        inf_sort = np.argsort(inf_x)

        # x smaller = left side, x larger = right side
        tl = box[superior_idx[sup_sort[0]]]
        tr = box[superior_idx[sup_sort[1]]]
        bl = box[inferior_idx[inf_sort[0]]]
        br = box[inferior_idx[inf_sort[1]]]

        result.extend([tl, tr, bl, br])
    return np.asarray(result, np.float32)


def extract_vertebra_boxes(pts: np.ndarray):
    """Extract 7 vertebral bodies (C2-T1) from 28 front points."""
    n = pts.shape[0]
    assert n == 28, f"Expected 28 points, got {n}"
    return [pts[i*4:(i+1)*4] for i in range(7)]


def _centers_from_boxes(boxes):
    """Compute centroid of each box."""
    return [b.mean(axis=0) for b in boxes]


def _endplate_angles(box: np.ndarray):
    """Compute upper and lower endplate angles from 4 corners [tl, tr, bl, br].
    Returns signed angles in (-180, 180]. Positive = tilting down to the right.
    """
    tl, tr, bl, br = box
    upper = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    lower = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    return float(upper), float(lower)


def compute_c2c7_lordosis(pts: np.ndarray) -> float:
    """Signed Cobb angle between C2 lower endplate and C7 lower endplate.
    Positive = lordosis (anterior convexity), Negative = kyphotic/reversed.
    """
    boxes = extract_vertebra_boxes(pts)
    _, c2_lower = _endplate_angles(boxes[0])
    _, c7_lower = _endplate_angles(boxes[5])

    diff = c2_lower - c7_lower
    if diff > 180:
        diff -= 360
    elif diff <= -180:
        diff += 360
    return float(diff)


def compute_max_cobb(pts: np.ndarray) -> float:
    """Maximum inter-vertebra Cobb angle among C2-C7.
    Upper vertebra uses upper endplate, lower vertebra uses lower endplate."""
    boxes = extract_vertebra_boxes(pts)
    endplates = [_endplate_angles(boxes[i]) for i in range(6)]  # C2-C7

    max_angle = 0
    for i in range(len(endplates)):
        for j in range(i + 1, len(endplates)):
            # i is above j: i uses upper, j uses lower
            diff = abs(endplates[i][0] - endplates[j][1])
            if diff > 90:
                diff = 180 - diff
            max_angle = max(max_angle, diff)
    return max_angle


def diagnose(cobb_angle: float) -> str:
    if cobb_angle < -10:
        return "反弓 (Kyphotic/Reversed)"
    elif cobb_angle < 0:
        return "轻度反弓/变直 (Mildly reversed)"
    elif cobb_angle < 10:
        return "严重变直 (Severely straightened)"
    elif cobb_angle < 20:
        return "变直 (Straightened)"
    elif cobb_angle < 40:
        return "正常 (Normal)"
    else:
        return "过度前凸 (Hyperlordosis)"


from PIL import Image, ImageDraw, ImageFont

def _draw_angle_line(img, cx, cy, angle_deg, length, color, thickness=2):
    """Draw a line from center (cx,cy) along angle_deg direction."""
    rad = np.radians(angle_deg)
    dx = np.cos(rad) * length
    dy = np.sin(rad) * length
    pt1 = (int(cx - dx), int(cy - dy))
    pt2 = (int(cx + dx), int(cy + dy))
    cv2.line(img, pt1, pt2, color, thickness)


def draw_cobb(image: np.ndarray, pts: np.ndarray):
    """Draw vertebrae boxes + C2/C7 Cobb reference lines (center-line method)."""
    vis = image.copy()
    boxes = extract_vertebra_boxes(pts)
    labels = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
              (255, 0, 255), (255, 255, 0)]

    centers = [b.mean(axis=0) for b in boxes]

    # Draw boxes (C2-C7 only, skip T1)
    for i, (box, label, color) in enumerate(zip(boxes[:6], labels, colors)):
        quad = box.astype(int)
        for j in range(4):
            cv2.circle(vis, tuple(quad[j]), 5, color, -1, lineType=cv2.LINE_AA)
        for a, b in [(0,1), (1,3), (3,2), (2,0)]:
            cv2.line(vis, tuple(quad[a]), tuple(quad[b]), color, 2, lineType=cv2.LINE_AA)
        cx, cy = centers[i]
        cv2.putText(vis, label, (int(cx)-15, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    # C2-C7 Cobb perpendicular reference lines (orange, thick)
    _, c2_lower = _endplate_angles(boxes[0])
    _, c7_lower = _endplate_angles(boxes[5])
    c2_bl, c2_br = boxes[0][2], boxes[0][3]
    c7_bl, c7_br = boxes[5][2], boxes[5][3]
    c2_mx, c2_my = (c2_bl + c2_br) / 2
    c7_mx, c7_my = (c7_bl + c7_br) / 2

    img_h, img_w = vis.shape[:2]
    diag_len = np.hypot(img_w, img_h)
    # Draw lines perpendicular to the lower endplates
    _draw_angle_line(vis, c2_mx, c2_my, c2_lower + 90, diag_len * 0.45, (0, 165, 255), 3)
    _draw_angle_line(vis, c7_mx, c7_my, c7_lower + 90, diag_len * 0.45, (0, 165, 255), 3)

    # Label angle near intersection of the two perpendiculars
    rad2, rad7 = np.radians(c2_lower), np.radians(c7_lower)
    # Perpendicular direction vectors: (-sin(theta), cos(theta))
    d2 = np.array([-np.sin(rad2), np.cos(rad2)])
    d7 = np.array([-np.sin(rad7), np.cos(rad7)])
    A = np.array([[d2[0], -d7[0]], [d2[1], -d7[1]]])
    b = np.array([c7_mx - c2_mx, c7_my - c2_my])
    # Text info
    lordosis = compute_c2c7_lordosis(pts)
    max_cobb = compute_max_cobb(pts)
    diag = diagnose(lordosis)

    try:
        t = np.linalg.solve(A, b)
        ix, iy = int(c2_mx + t[0]*d2[0]), int(c2_my + t[0]*d2[1])
        cv2.putText(vis, f"Cobb={lordosis:.1f}°", (ix-40, iy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    except np.linalg.LinAlgError:
        pass
    
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(vis_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    font_paths = ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/simsun.ttc']
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 24)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    
    texts = [f"C2-C7 前凸角: {lordosis:.1f}°", f"最大 Cobb 角: {max_cobb:.1f}°", f"诊断: {diag}"]
    for i, text in enumerate(texts):
        draw.text((10, 10 + i*30), text, fill=(255, 0, 0), font=font)
    
    vis = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return vis

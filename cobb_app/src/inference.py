"""
HRNet + VLD Fusion inference for single image.
Stable version: stats-based center/scale estimation + VLD vertebra selection.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
import torch
import torch.nn as nn

from hrnet_lib.core.evaluation import decode_preds
from hrnet_lib.utils.transforms import crop
from hrnet_lib.config import config, update_config
from hrnet_lib import models

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _bbox_center_scale_from_image(h, w):
    """Estimate center and scale from image size (no GT points).

    Training stats from RENJI data (n=183):
      - center_x offset from image center: median = -3.2%
      - center_y offset from image center: median = +5.1%
      - scale / whole_image_scale:         median = 0.483
    """
    center_x = w * 0.468
    center_y = h * 0.551
    scale = max(w, h) * 0.48 / 200.0 * 1.25
    return np.array([center_x, center_y], dtype=np.float32), float(scale)


def preprocess_image(image_path, input_size=(256, 256)):
    """Read and preprocess image for HRNet."""
    img_array = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = image.shape[:2]
    center, scale = _bbox_center_scale_from_image(h, w)

    center_t = torch.from_numpy(center)
    out_image = crop(image, center_t, scale, input_size, rot=0)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out_image = out_image.astype(np.float32)
    out_image = (out_image / 255.0 - mean) / std
    out_image = out_image.transpose([2, 0, 1])

    tensor = torch.from_numpy(out_image).unsqueeze(0)
    return tensor, center, scale, image


def load_model(cfg_path, model_path, device='cuda'):
    """Load HRNet model."""
    import hrnet_lib
    if 'lib' not in sys.modules:
        sys.modules['lib'] = hrnet_lib
        sys.modules['lib.models'] = hrnet_lib.models
        sys.modules['lib.core'] = hrnet_lib.core
        sys.modules['lib.utils'] = hrnet_lib.utils
        sys.modules['lib.config'] = hrnet_lib.config

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--model-file', type=str, default='')
    parser.add_argument('--resolution-csv', type=str, default='')
    parser.add_argument('--default-resolution', type=float, default=1.0)
    parser.add_argument('--matching-mode', type=str, default='hungarian')
    args = parser.parse_args([])
    args.cfg = cfg_path

    update_config(config, args)
    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    model = models.get_face_alignment_net(config)

    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict.state_dict())

    if device == 'cuda' and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[0]).cuda()
    else:
        model = model.cpu()

    model.eval()
    return model, config


def predict(model, config, image_path, device='cuda'):
    """Run HRNet inference on a single image.
    Returns all 56 points (front 28 + back 28)."""
    tensor, center, scale, orig_image = preprocess_image(image_path)

    if device == 'cuda' and torch.cuda.is_available():
        tensor = tensor.cuda()

    with torch.no_grad():
        output = model(tensor)

    score_map = output.data.cpu()
    heatmap_size = config.MODEL.HEATMAP_SIZE

    center_t = torch.from_numpy(center).unsqueeze(0)
    scale_t = torch.tensor([scale])
    preds = decode_preds(score_map, center_t, scale_t, heatmap_size)

    pts56 = preds[0].cpu().numpy()  # (56, 2)
    return pts56, orig_image


def _vertebra_centers(corners):
    """corners: (N*4, 2) -> centers: (N, 2)"""
    n = len(corners) // 4
    return np.array([corners[i*4:(i+1)*4].mean(axis=0) for i in range(n)])


def _bbox_aspect_ratios(corners):
    """Compute width/height aspect ratio for each vertebra from 4 corners.
    corners: (N*4, 2) where each 4 pts are [tl, tr, bl, br].
    Returns: (N,) array of aspect ratios.
    """
    n = len(corners) // 4
    ratios = []
    for i in range(n):
        box = corners[i*4:(i+1)*4]
        xs = box[:, 0]
        ys = box[:, 1]
        width = xs.max() - xs.min()
        height = ys.max() - ys.min()
        ratios.append(width / max(height, 1e-6))
    return np.array(ratios)


def fuse_hrnet_vld(hrnet_56pts, vld_pts, back_penalty=150.0):
    """Select best 7 vertebrae (C2-T1) from VLD using HRNet front/back centers.
    
    All VLD candidates compete for 14 anchors (7 front + 7 back).
    Posterior candidates (closer to back) get penalized in front rows,
    so spinous processes match to "back" instead of stealing front vertebrae.
    Missing front matches fall back to HRNet predictions.
    
    Args:
        hrnet_56pts: (56, 2) - full HRNet output (front 28 + back 28).
        vld_pts: (N*4, 2) - N vertebrae x 4 corners, grouped by 4, sorted top-to-bottom.
    
    Returns:
        (28, 2) - selected 7 vertebrae ordered C2..T1.
    """
    if linear_sum_assignment is None:
        raise ImportError('scipy is required for HRNet-VLD fusion (pip install scipy)')

    hrnet_front = hrnet_56pts[:28]   # C2-T1 front
    hrnet_back = hrnet_56pts[28:]    # C2-T1 back
    
    vld_boxes = vld_pts.reshape(-1, 4, 2)
    vld_c = vld_boxes.mean(axis=1)
    
    hrnet_front_c = np.array([hrnet_front[i*4:(i+1)*4].mean(axis=0) for i in range(7)])
    hrnet_back_c = np.array([hrnet_back[i*4:(i+1)*4].mean(axis=0) for i in range(7)])

    anchors = np.vstack([hrnet_front_c, hrnet_back_c])  # (14, 2)
    cost = np.linalg.norm(anchors[:, None, :] - vld_c[None, :, :], axis=2)

    for v in range(vld_c.shape[0]):
        if cost[7:, v].min() < cost[:7, v].min():
            cost[:7, v] += back_penalty

    row_ind, col_ind = linear_sum_assignment(cost)

    matched = []
    for i in range(7):  # front anchors only
        if i in row_ind:
            vidx = col_ind[np.where(row_ind == i)[0][0]]
            matched.extend(vld_boxes[vidx].tolist())
        else:
            matched.extend(hrnet_front[i*4:(i+1)*4].tolist())
    return np.asarray(matched, np.float32)


def predict_fusion(hrnet_model, config, vld_model, image_path, device='cuda',
                   fusion_weight=1.0):
    """Run HRNet + VLD fusion inference on a single image.
    Unifies orientation so that vertebrae (front) are always on the left.
    
    Args:
        fusion_weight: float in [0, 1]. 1.0 = pure VLD (selected by HRNet),
                       0.0 = pure HRNet, 0.5 = average.
    
    Returns:
        (pts_28, orig_image) where pts_28 is (28, 2) numpy array.
    """
    # HRNet prediction (56 pts = front 28 + back 28)
    hrnet_56pts, orig_image = predict(hrnet_model, config, image_path, device)

    # VLD prediction
    vld_pts, _ = vld_model.predict(image_path)

    # Fuse: select 7 VLD vertebrae guided by HRNet front/back centers
    fusion_pts = fuse_hrnet_vld(hrnet_56pts, vld_pts)

    if fusion_weight < 1.0:
        hrnet_front = hrnet_56pts[:28]
        fusion_pts = fusion_weight * fusion_pts + (1.0 - fusion_weight) * hrnet_front

    # --- Unify orientation: front (vertebrae) should be on the left ---
    hrnet_front = hrnet_56pts[:28]
    hrnet_back = hrnet_56pts[28:]
    front_x = np.median(hrnet_front[:, 0])
    back_x = np.median(hrnet_back[:, 0])

    if front_x > back_x:
        # Front is on the right: flip image and all points horizontally
        h, w = orig_image.shape[:2]
        orig_image = cv2.flip(orig_image, 1)
        fusion_pts[:, 0] = w - 1 - fusion_pts[:, 0]
        hrnet_56pts[:, 0] = w - 1 - hrnet_56pts[:, 0]

    return fusion_pts, orig_image

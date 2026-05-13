"""
VLD (Vertebra-Landmark-Detection) inference wrapper for cobb_app.
Loads SpineNet model and runs single-image inference with TTA.
"""
import sys
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Add VLD repo to path
_VLD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Vertebra-Landmark-Detection'))
if _VLD_DIR not in sys.path:
    sys.path.insert(0, _VLD_DIR)

from models import spinal_net
import decoder
import pre_proc
import transform


def inverse_letterbox_coords(pts, src_w, src_h, input_w, input_h):
    """Map coordinates from letterboxed input back to original image."""
    scale = min(input_w / float(src_w), input_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    pad_x = (input_w - new_w) // 2
    pad_y = (input_h - new_h) // 2
    x_index = range(0, 10, 2)
    y_index = range(1, 10, 2)
    pts[:, x_index] = (pts[:, x_index] - pad_x) / scale
    pts[:, y_index] = (pts[:, y_index] - pad_y) / scale
    return pts


def rearrange_pts(pts):
    """Rearrange 4-corners per vertebra: tl, tr, bl, br. Sort vertebrae top-to-bottom.
    Uses adjacent spine segment as local axis for robust ordering under flexion.
    """
    n = len(pts) // 4
    raw_boxes = [pts[k:k+4].astype(np.float32) for k in range(0, len(pts), 4)]
    raw_centers = [b.mean(axis=0) for b in raw_boxes]

    # Sort vertebrae top-to-bottom by center y
    sort_idx = np.argsort([c[1] for c in raw_centers])

    sorted_boxes = []
    for rank, idx in enumerate(sort_idx):
        box = raw_boxes[idx]
        center = raw_centers[idx]

        # Local cranial-caudal axis: point to next vertebra (caudal)
        if rank < n - 1:
            axis_vec = raw_centers[sort_idx[rank + 1]] - center
        else:
            axis_vec = center - raw_centers[sort_idx[rank - 1]]

        # Ensure axis points roughly downward (positive y)
        if axis_vec[1] < 0:
            axis_vec = -axis_vec

        centered = box - center

        # Perpendicular axis (endplate direction): rotate 90 deg CCW
        perp = np.array([-axis_vec[1], axis_vec[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-6)
        axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-6)

        proj_axis = centered @ axis
        proj_perp = centered @ perp

        axis_sort = np.argsort(proj_axis)
        superior_idx = axis_sort[:2]
        inferior_idx = axis_sort[2:]

        sup_perp = proj_perp[superior_idx]
        inf_perp = proj_perp[inferior_idx]
        sup_sort = np.argsort(sup_perp)
        inf_sort = np.argsort(inf_perp)

        tl = box[superior_idx[sup_sort[1]]]
        tr = box[superior_idx[sup_sort[0]]]
        bl = box[inferior_idx[inf_sort[1]]]
        br = box[inferior_idx[inf_sort[0]]]

        sorted_boxes.extend([tl, tr, bl, br])

    return np.asarray(sorted_boxes, np.float32)


class VLDModel:
    """Wrapper for VLD SpineNet inference."""

    def __init__(self,
                 weights_path,
                 input_h=1024,
                 input_w=512,
                 down_ratio=4,
                 K=14,
                 conf_thresh=0.2,
                 num_classes=1,
                 use_tta=True,
                 device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.K = K
        self.use_tta = use_tta

        heads = {'hm': num_classes, 'reg': 2 * num_classes, 'wh': 2 * 4}
        self.model = spinal_net.SpineNet(
            heads=heads,
            pretrained=True,
            down_ratio=down_ratio,
            final_kernel=1,
            head_conv=256
        )
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
        print('loaded VLD weights from {}, epoch {}'.format(weights_path, checkpoint.get('epoch', 'unknown')))
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.decoder = decoder.DecDecoder(K=K, conf_thresh=conf_thresh)

    def preprocess(self, image_path):
        """Load image and apply VLD test preprocessing."""
        img_array = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        image_proc = pre_proc.processing_test(image, self.input_h, self.input_w)
        return image_proc, image

    def predict(self, image_path, conf_thresh=0.15):
        """Run VLD inference on a single image.
        Returns: (landmarks_56, orig_image)
            landmarks_56: np.ndarray (N*4, 2) in original image coordinates.
                          Grouped by vertebra (4 pts each), sorted top-to-bottom.
                          N may be <= K after confidence filtering.
        """
        image_proc, orig_image = self.preprocess(image_path)
        image_proc = image_proc.to(self.device)
        h, w = orig_image.shape[:2]

        with torch.no_grad():
            out = self.model(image_proc)
            hm = out['hm']
            wh = out['wh']
            reg = out['reg']
            forward_count = 1

            if self.use_tta:
                images_flipped = torch.flip(image_proc, dims=[3])
                out_f = self.model(images_flipped)
                hm_f = out_f['hm']
                hm_f = torch.flip(hm_f, dims=[3])
                hm = hm + hm_f
                forward_count = 2

            hm = hm / forward_count

        pts2 = self.decoder.ctdet_decode(hm, wh, reg)
        pts0 = pts2.copy()

        # ---- Confidence filtering ----
        scores = pts0[:, 10]
        keep = scores >= conf_thresh
        if keep.sum() < 7:
            # Not enough candidates after filtering; take top 7 by score
            top7 = np.argsort(scores)[::-1][:7]
            keep = np.zeros(len(pts0), dtype=bool)
            keep[top7] = True
        pts0 = pts0[keep]

        pts0[:, :10] *= self.down_ratio
        pts0 = inverse_letterbox_coords(pts0, w, h, self.input_w, self.input_h)

        # sort by y (top to bottom)
        sort_ind = np.argsort(pts0[:, 1])
        pts0 = pts0[sort_ind]

        # extract 4 corners per vertebra
        pr_landmarks = []
        for pt in pts0:
            pr_landmarks.append(pt[2:4])
            pr_landmarks.append(pt[4:6])
            pr_landmarks.append(pt[6:8])
            pr_landmarks.append(pt[8:10])
        pr_landmarks = np.asarray(pr_landmarks, np.float32)

        # rearrange corners within each vertebra
        pr_landmarks = rearrange_pts(pr_landmarks)

        return pr_landmarks, orig_image

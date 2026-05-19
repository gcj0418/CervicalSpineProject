"""
Split-VLD inference: vertebrae model (28 pts) + facets model (28 pts).
No HRNet, no Hungarian matching. Each model outputs 7 boxes sorted by y,
directly corresponding to C2..T1. Concatenate to 56 pts.

Supports PNG/JPG and NIfTI (.nii/.nii.gz) inputs.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Vertebra-Landmark-Detection'))

import numpy as np
import cv2
import torch

from pre_proc import processing_test
from decoder import DecDecoder
from models import spinal_net


def load_medical_image(image_path):
    """Load PNG/JPG or NIfTI and return BGR uint8 image + optional voxel spacing."""
    path = str(image_path)
    lower = path.lower()
    if lower.endswith('.nii.gz') or lower.endswith('.nii'):
        try:
            import nibabel as nib
        except ImportError as e:
            raise ImportError(
                "nibabel is required to load NIfTI files. "
                "Install it via: pip install nibabel"
            ) from e
        nii = nib.load(path)
        data = nii.get_fdata().astype(np.float32)

        if data.ndim == 2:
            img2d = data
        elif data.ndim == 3:
            if data.shape[2] == 1:
                img2d = data[:, :, 0]
            else:
                z = data.shape[2] // 2
                img2d = data[:, :, z]
        else:
            raise ValueError(f"Unsupported NIfTI shape: {data.shape}")

        img2d = np.ascontiguousarray(np.rot90(img2d, k=3))
        finite = img2d[np.isfinite(img2d)]
        lo = float(np.percentile(finite, 2))
        hi = float(np.percentile(finite, 98))
        if hi <= lo:
            hi = lo + 1.0
        img2d = np.clip(img2d, lo, hi)
        img2d = (img2d - lo) / (hi - lo + 1e-8)
        img_u8 = (img2d * 255.0).round().astype(np.uint8)
        image = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

        affine = nii.affine
        sx = abs(float(affine[0, 0]))
        sy = abs(float(affine[1, 1]))
        spacing = (sx + sy) / 2.0
        return image, spacing
    else:
        img_array = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot load image: {path}")
        return image, None


class SplitVLDModel:
    """Wrapper for split VLD inference (vertebrae + facets)."""

    def __init__(self,
                 vert_weights_path,
                 facet_weights_path,
                 input_h=1024,
                 input_w=512,
                 down_ratio=4,
                 K=7,
                 conf_thresh=0.2,
                 use_tta=True,
                 device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.K = K
        self.conf_thresh = conf_thresh
        self.use_tta = use_tta

        heads = {'hm': 1, 'reg': 2, 'wh': 8}

        # Load vertebrae model
        self.vert_model = spinal_net.SpineNet(
            heads=heads, pretrained=True, down_ratio=down_ratio,
            final_kernel=1, head_conv=256)
        cp = torch.load(vert_weights_path, map_location=lambda storage, loc: storage)
        self.vert_model.load_state_dict(cp['state_dict'], strict=False)
        self.vert_model.to(self.device)
        self.vert_model.eval()
        self.vert_decoder = DecDecoder(K=K, conf_thresh=conf_thresh)

        # Load facets model
        self.facet_model = spinal_net.SpineNet(
            heads=heads, pretrained=True, down_ratio=down_ratio,
            final_kernel=1, head_conv=256)
        cp = torch.load(facet_weights_path, map_location=lambda storage, loc: storage)
        self.facet_model.load_state_dict(cp['state_dict'], strict=False)
        self.facet_model.to(self.device)
        self.facet_model.eval()
        self.facet_decoder = DecDecoder(K=K, conf_thresh=conf_thresh)

    def _infer_single(self, model, decoder, image_proc, w, h):
        """Run one model, return landmarks (N*4, 2) sorted top-to-bottom."""
        with torch.no_grad():
            out = model(image_proc)
            hm = out['hm']
            wh = out['wh']
            reg = out['reg']
            forward_count = 1

            if self.use_tta:
                images_flipped = torch.flip(image_proc, dims=[3])
                out_f = model(images_flipped)
                hm_f = out_f['hm']
                hm_f = torch.flip(hm_f, dims=[3])
                hm = hm + hm_f
                forward_count = 2

            hm = hm / forward_count

        pts2 = decoder.ctdet_decode(hm, wh, reg)
        pts0 = pts2.copy()

        # Confidence filtering (same logic as original VLDModel)
        scores = pts0[:, 10]
        keep = scores >= self.conf_thresh
        if keep.sum() < self.K:
            topk = np.argsort(scores)[::-1][:self.K]
            keep = np.zeros(len(pts0), dtype=bool)
            keep[topk] = True
        pts0 = pts0[keep]

        pts0[:, :10] *= self.down_ratio

        # Inverse letterbox
        scale = min(self.input_w / float(w), self.input_h / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2
        x_idx = list(range(0, 10, 2))
        y_idx = list(range(1, 10, 2))
        pts0[:, x_idx] = (pts0[:, x_idx] - pad_x) / scale
        pts0[:, y_idx] = (pts0[:, y_idx] - pad_y) / scale

        # Sort by y (top-to-bottom = C2..T1)
        sort_ind = np.argsort(pts0[:, 1])
        pts0 = pts0[sort_ind]

        # Extract 4 corners: [tl, tr, bl, br]
        landmarks = []
        for pt in pts0:
            landmarks.append(pt[2:4])   # tl
            landmarks.append(pt[4:6])   # tr
            landmarks.append(pt[6:8])   # bl
            landmarks.append(pt[8:10])  # br
        return np.asarray(landmarks, np.float32)

    def predict(self, image_path):
        """Run both models and return (pts_front, pts_back, orig_image, spacing).

        pts_front: vertebrae 28 pts (C2..T1), from vert model.
        pts_back:  facets 28 pts (C2..T1), from facet model.
        """
        image, spacing = load_medical_image(image_path)
        h, w = image.shape[:2]

        image_proc = processing_test(image, self.input_h, self.input_w)
        image_proc = image_proc.to(self.device)

        vert_pts = self._infer_single(self.vert_model, self.vert_decoder, image_proc, w, h)
        facet_pts = self._infer_single(self.facet_model, self.facet_decoder, image_proc, w, h)

        # Direction unify: vertebrae (front) should be on the left
        vert_median_x = np.median(vert_pts[:, 0])
        facet_median_x = np.median(facet_pts[:, 0])

        if vert_median_x > facet_median_x:
            # Vertebrae on the right -> flip horizontally
            image = cv2.flip(image, 1)
            vert_pts[:, 0] = w - 1 - vert_pts[:, 0]
            facet_pts[:, 0] = w - 1 - facet_pts[:, 0]
            # Swap tl <-> tr, bl <-> br within each box
            for pts in (vert_pts, facet_pts):
                n = pts.shape[0] // 4
                for i in range(n):
                    base = i * 4
                    pts[[base, base + 1]] = pts[[base + 1, base]]
                    pts[[base + 2, base + 3]] = pts[[base + 3, base + 2]]

        return vert_pts, facet_pts, image, spacing


def predict_fusion_legacy(*args, **kwargs):
    """Deprecated: old HRNet+VLD fusion. Kept for reference."""
    raise NotImplementedError("HRNet+VLD fusion removed. Use SplitVLDModel.predict().")

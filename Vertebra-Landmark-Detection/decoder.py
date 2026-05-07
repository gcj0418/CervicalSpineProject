import torch.nn.functional as F
import numpy as np
import torch

class DecDecoder(object):
    def __init__(self, K, conf_thresh):
        self.K = K
        self.conf_thresh = conf_thresh

    def _topk(self, scores):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_ys, topk_xs


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def ctdet_decode(self, heat, wh, reg):
        # output: num_obj x 7
        # 7: cenx, ceny, w, h, angle, score, cls
        batch, c, height, width = heat.size()
        heat = self._nms(heat)   # [1, 1, 256, 128]
        scores, inds, ys, xs = self._topk(heat)
        scores = scores.view(batch, self.K, 1)
        reg = self._tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, self.K, 2*4)

        tl_x = xs - wh[:,:,0:1]
        tl_y = ys - wh[:,:,1:2]
        tr_x = xs - wh[:,:,2:3]
        tr_y = ys - wh[:,:,3:4]
        bl_x = xs - wh[:,:,4:5]
        bl_y = ys - wh[:,:,5:6]
        br_x = xs - wh[:,:,6:7]
        br_y = ys - wh[:,:,7:8]

        # subpixel refinement: use local 3x3 patch centroid on the (NMSed) heatmap
        try:
            heat_np = heat.detach().cpu().numpy()
            _, _, H, W = heat_np.shape
            for b in range(batch):
                for k in range(self.K):
                    x_val = xs[b, k, 0].item()
                    y_val = ys[b, k, 0].item()
                    ix = int(round(x_val))
                    iy = int(round(y_val))
                    ix0 = max(ix - 1, 0)
                    ix1 = min(ix + 1, W - 1)
                    iy0 = max(iy - 1, 0)
                    iy1 = min(iy + 1, H - 1)
                    patch = heat_np[b, 0, iy0:iy1+1, ix0:ix1+1]
                    if patch.size == 0:
                        continue
                    ys_idx = np.arange(iy0, iy1+1)
                    xs_idx = np.arange(ix0, ix1+1)
                    xx, yy = np.meshgrid(xs_idx, ys_idx)
                    wsum = patch.sum() + 1e-8
                    cx = (patch * xx).sum() / wsum
                    cy = (patch * yy).sum() / wsum
                    dx = cx - x_val
                    dy = cy - y_val
                    xs[b, k, 0] = xs[b, k, 0] + float(dx)
                    ys[b, k, 0] = ys[b, k, 0] + float(dy)
        except Exception:
            # if any error, skip refinement and continue
            pass

        pts = torch.cat([xs, ys,
                         tl_x,tl_y,
                         tr_x,tr_y,
                         bl_x,bl_y,
                         br_x,br_y,
                         scores], dim=2).squeeze(0)
        return pts.data.cpu().numpy()
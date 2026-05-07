import torch
import torch.nn.functional as F
import numpy as np
import csv
from models import spinal_net
import decoder
import os
from dataset import BaseDataset
import time
import cobb_evaluate

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def load_resolution_map(path):
    if not path:
        return {}
    resolution_map = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[0].strip()
            try:
                if len(row) >= 3:
                    resolution_map[name] = np.array([float(row[1]), float(row[2])], dtype=np.float32)
                else:
                    resolution_map[name] = float(row[1])
            except ValueError:
                continue
    return resolution_map


def get_sample_resolution(resolution_map, img_id, default_resolution):
    sample_key = os.path.splitext(img_id)[0]
    if img_id in resolution_map:
        return resolution_map[img_id]
    if sample_key in resolution_map:
        return resolution_map[sample_key]
    return default_resolution


def point_error_mm(pred_point, gt_point, resolution):
    diff = np.asarray(pred_point, dtype=np.float32) - np.asarray(gt_point, dtype=np.float32)
    resov_arr = np.asarray(resolution, dtype=np.float32).reshape(-1)
    if resov_arr.size >= 2:
        return np.sqrt((diff[0] * resov_arr[0]) ** 2 + (diff[1] * resov_arr[1]) ** 2)
    return np.sqrt(np.sum(np.power(diff, 2))) * float(resov_arr[0] if resov_arr.size else 1.0)


def pair_landmarks(pr_landmarks, gt_landmarks, resolution, matching_mode='hungarian'):
    pr_landmarks = np.asarray(pr_landmarks, dtype=np.float32)
    gt_landmarks = np.asarray(gt_landmarks, dtype=np.float32)
    if len(pr_landmarks) == 0 or len(gt_landmarks) == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty, np.asarray([], dtype=np.float32)

    if matching_mode == 'hungarian':
        if linear_sum_assignment is None:
            raise ImportError('matching_mode=hungarian requires scipy (pip install scipy)')

        resov_arr = np.asarray(resolution, dtype=np.float32).reshape(-1)
        if resov_arr.size >= 2:
            spacing = np.array([resov_arr[0], resov_arr[1]], dtype=np.float32)
        else:
            scale = float(resov_arr[0] if resov_arr.size else 1.0)
            spacing = np.array([scale, scale], dtype=np.float32)

        pr_scaled = pr_landmarks * spacing
        gt_scaled = gt_landmarks * spacing
        cost = np.linalg.norm(pr_scaled[:, None, :] - gt_scaled[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        pr_matched = pr_landmarks[row_ind]
        gt_matched = gt_landmarks[col_ind]
        dists = cost[row_ind, col_ind].astype(np.float32)
        return pr_matched, gt_matched, dists

    raise ValueError('Only matching_mode=hungarian is supported, got {}'.format(matching_mode))


def inverse_letterbox_coords(pts, src_w, src_h, input_w, input_h):
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

def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,  # cen, tl, tr, bl, br
                 'reg': 2*args.num_classes,
                 'wh': 2*4,}

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'spinal': BaseDataset, 'ruijin': BaseDataset, 'renji': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def instantiate_model(self, args):
        heads = {'hm': args.num_classes,  # cen, tl, tr, bl, br
                 'reg': 2*args.num_classes,
                 'wh': 2*4,}
        model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256)
        return model


    def eval(self, args, save):
        save_path = 'weights_'+args.dataset
        # support multiple checkpoints separated by comma
        resumes = [r.strip() for r in args.resume.split(',')] if ',' in args.resume else [args.resume]
        models = []
        for r in resumes:
            m = self.instantiate_model(args)
            checkpoint = torch.load(os.path.join(save_path, r), map_location=lambda storage, loc: storage)
            print('loaded weights from {}, epoch {}'.format(r, checkpoint.get('epoch', 'unknown')))
            state_dict_ = checkpoint['state_dict']
            m.load_state_dict(state_dict_, strict=False)
            m = m.to(self.device)
            m.eval()
            models.append(m)
        self.models = models
        resolution_map = load_resolution_map(getattr(args, 'resolution_csv', ''))
        matching_mode = getattr(args, 'matching_mode', 'hungarian')
        print('landmark matching mode: {}'.format(matching_mode))
        eval_phase = getattr(args, 'eval_phase', 'test')
        dump_sample_csv = getattr(args, 'dump_sample_csv', '')
        per_sample_rows = []

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                       phase=eval_phase,
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio,
                               max_points=args.max_points,
                               augment=False)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)

        total_time = []
        landmark_dist = []
        pr_cobb_angles = []
        gt_cobb_angles = []
        all_predictions = {}
        for cnt, data_dict in enumerate(data_loader):
            begin_time = time.time()
            if 'images' in data_dict:
                images = data_dict['images'][0]
                img_id = data_dict['img_id'][0]
            else:
                images = data_dict['input']
                img_id = dsets.img_ids[cnt]
            images = images.to(self.device)
            print('processing {}/{} image ...'.format(cnt, len(data_loader)))

            # Ensemble heatmaps across checkpoints and TTA (horizontal flip)
            hm_sum = None
            wh = None
            reg = None
            forward_count = 0
            with torch.no_grad():
                for mi, m in enumerate(self.models):
                    out = m(images)
                    hm_i = out['hm']
                    if hm_sum is None:
                        hm_sum = hm_i.clone()
                    else:
                        hm_sum += hm_i
                    # keep wh/reg from first model's original pass
                    if mi == 0:
                        wh = out['wh']
                        reg = out['reg']
                    forward_count += 1
                    # TTA: horizontal flip
                    if getattr(args, 'tta', False):
                        images_flipped = torch.flip(images, dims=[3])
                        out_f = m(images_flipped)
                        hm_f = out_f['hm']
                        # flip heatmap back
                        hm_f = torch.flip(hm_f, dims=[3])
                        hm_sum += hm_f
                        forward_count += 1
            # average heatmap
            hm = hm_sum / max(forward_count, 1)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            # optionally upsample heatmap/reg/wh to increase decoding resolution
            upsample = getattr(args, 'upsample', 1)
            effective_down = args.down_ratio
            if upsample is not None and int(upsample) > 1:
                scale = int(upsample)
                hm = F.interpolate(hm, scale_factor=scale, mode='bilinear', align_corners=False)
                wh = F.interpolate(wh, scale_factor=scale, mode='bilinear', align_corners=False)
                reg = F.interpolate(reg, scale_factor=scale, mode='bilinear', align_corners=False)
                effective_down = args.down_ratio / scale

            pts2 = self.decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            pts0 = pts2.copy()
            pts0[:,:10] *= effective_down
            ori_image = dsets.load_image(dsets.img_ids.index(img_id)).copy()
            h,w,c = ori_image.shape
            pts0 = inverse_letterbox_coords(pts0, w, h, args.input_w, args.input_h)
            # sort the y axis
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]
            pr_landmarks = []
            for i, pt in enumerate(pts0):
                pr_landmarks.append(pt[2:4])
                pr_landmarks.append(pt[4:6])
                pr_landmarks.append(pt[6:8])
                pr_landmarks.append(pt[8:10])
            pr_landmarks = np.asarray(pr_landmarks, np.float32)   #[68, 2]

            end_time = time.time()
            total_time.append(end_time-begin_time)

            gt_landmarks = dsets.load_gt_pts(dsets.load_annoFolder(img_id))
            sample_resolution = get_sample_resolution(
                resolution_map,
                img_id,
                getattr(args, 'default_resolution', 1.0),
            )
            _, _, matched_dists = pair_landmarks(
                pr_landmarks,
                gt_landmarks,
                sample_resolution,
                matching_mode=matching_mode,
            )
            landmark_dist.extend(matched_dists.tolist())
            if dump_sample_csv:
                sample_error = float(np.mean(matched_dists)) if len(matched_dists) else float('nan')
                per_sample_rows.append((img_id, sample_error))

            all_predictions[img_id] = pr_landmarks.copy()

            pr_cobb_angles.append(cobb_evaluate.cobb_angle_calc(pr_landmarks, ori_image))
            gt_cobb_angles.append(cobb_evaluate.cobb_angle_calc(gt_landmarks, ori_image))

        pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
        gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)

        out_abs = abs(gt_cobb_angles - pr_cobb_angles)
        out_add = gt_cobb_angles + pr_cobb_angles

        term1 = np.sum(out_abs, axis=1)
        term2 = np.sum(out_add, axis=1)

        SMAPE = np.mean(term1 / term2 * 100)

        landmark_dist = np.asarray(landmark_dist, np.float32)
        total_points = max(len(landmark_dist), 1)
        print('2mm acc = {}'.format(np.sum(landmark_dist < 2.0) / total_points))
        print('2.5mm acc = {}'.format(np.sum(landmark_dist < 2.5) / total_points))
        print('3mm acc = {}'.format(np.sum(landmark_dist < 3.0) / total_points))
        print('4mm acc = {}'.format(np.sum(landmark_dist < 4.0) / total_points))
        print('mean landmark error (mm) is {}'.format(np.mean(landmark_dist)))
        print('SMAPE is {}'.format(SMAPE))

        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1./np.mean(total_time)))

        # Write comparison_table.md (unified schema with HRNet / D-CeLR)
        output_base = getattr(args, 'output_dir', 'outputs')
        logs_dir = os.path.join(output_base, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        comparison_md = os.path.join(logs_dir, 'comparison_table.md')
        with open(comparison_md, 'w', encoding='utf-8') as f:
            f.write('| dataset | model | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |\n')
            f.write('| --- | --- | ---: | ---: | ---: | ---: | ---: |\n')
            f.write(
                '| {dataset} | {model} | {mean:.4f} | {a2:.4f} | {a25:.4f} | {a3:.4f} | {a4:.4f} |\n'.format(
                    dataset=getattr(args, 'dataset', 'unknown'),
                    model=getattr(args, 'resume', 'model_last.pth'),
                    mean=float(np.mean(landmark_dist)),
                    a2=float(np.sum(landmark_dist < 2.0) / total_points),
                    a25=float(np.sum(landmark_dist < 2.5) / total_points),
                    a3=float(np.sum(landmark_dist < 3.0) / total_points),
                    a4=float(np.sum(landmark_dist < 4.0) / total_points),
                )
            )
        print('saved comparison table to {}'.format(comparison_md))

        # Save predictions for ensemble
        preds_dir = os.path.join(output_base, 'predictions')
        os.makedirs(preds_dir, exist_ok=True)
        torch.save(all_predictions, os.path.join(preds_dir, 'predictions.pth'))
        print('saved predictions to {}'.format(os.path.join(preds_dir, 'predictions.pth')))

        if dump_sample_csv:
            valid_errors = np.asarray([row[1] for row in per_sample_rows if np.isfinite(row[1])], dtype=np.float32)
            mean_error = float(np.mean(valid_errors)) if valid_errors.size else 1.0
            with open(dump_sample_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['img_id', 'mean_error_mm', 'sample_weight'])
                for img_id, sample_error in per_sample_rows:
                    if not np.isfinite(sample_error):
                        weight = 1.0
                    else:
                        weight = float(sample_error / (mean_error + 1e-6))
                    writer.writerow([img_id, sample_error, weight])
            print('saved per-sample errors to {}'.format(dump_sample_csv))


    def SMAPE_single_angle(self, gt_cobb_angles, pr_cobb_angles):
        out_abs = abs(gt_cobb_angles - pr_cobb_angles)
        out_add = gt_cobb_angles + pr_cobb_angles

        term1 = out_abs
        term2 = out_add

        term2[term2==0] += 1e-5

        SMAPE = np.mean(term1 / term2 * 100)
        return SMAPE

    def eval_three_angles(self, args, save):
        save_path = 'weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()
        resolution_map = load_resolution_map(getattr(args, 'resolution_csv', ''))
        matching_mode = getattr(args, 'matching_mode', 'hungarian')
        print('landmark matching mode: {}'.format(matching_mode))
        eval_phase = getattr(args, 'eval_phase', 'test')
        dump_sample_csv = getattr(args, 'dump_sample_csv', '')
        per_sample_rows = []

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                       phase=eval_phase,
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio,
                       max_points=args.max_points,
                       augment=False)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)

        total_time = []
        landmark_dist = []
        pr_cobb_angles = []
        gt_cobb_angles = []
        for cnt, data_dict in enumerate(data_loader):
            begin_time = time.time()
            if 'images' in data_dict:
                images = data_dict['images'][0]
                img_id = data_dict['img_id'][0]
            else:
                images = data_dict['input']
                img_id = dsets.img_ids[cnt]
            images = images.to(self.device)
            print('processing {}/{} image ...'.format(cnt, len(data_loader)))

            with torch.no_grad():
                output = self.model(images)
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio
            ori_image = dsets.load_image(dsets.img_ids.index(img_id)).copy()
            h,w,c = ori_image.shape
            pts0 = inverse_letterbox_coords(pts0, w, h, args.input_w, args.input_h)
            # sort the y axis
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]
            pr_landmarks = []
            for i, pt in enumerate(pts0):
                pr_landmarks.append(pt[2:4])
                pr_landmarks.append(pt[4:6])
                pr_landmarks.append(pt[6:8])
                pr_landmarks.append(pt[8:10])
            pr_landmarks = np.asarray(pr_landmarks, np.float32)   #[68, 2]

            end_time = time.time()
            total_time.append(end_time-begin_time)

            gt_landmarks = dsets.load_gt_pts(dsets.load_annoFolder(img_id))
            sample_resolution = get_sample_resolution(
                resolution_map,
                img_id,
                getattr(args, 'default_resolution', 1.0),
            )
            _, _, matched_dists = pair_landmarks(
                pr_landmarks,
                gt_landmarks,
                sample_resolution,
                matching_mode=matching_mode,
            )
            landmark_dist.extend(matched_dists.tolist())
            if dump_sample_csv:
                sample_error = float(np.mean(matched_dists)) if len(matched_dists) else float('nan')
                per_sample_rows.append((img_id, sample_error))

            pr_cobb_angles.append(cobb_evaluate.cobb_angle_calc(pr_landmarks, ori_image))
            gt_cobb_angles.append(cobb_evaluate.cobb_angle_calc(gt_landmarks, ori_image))

        pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
        gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)


        print('SMAPE1 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,0], pr_cobb_angles[:,0])))
        print('SMAPE2 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,1], pr_cobb_angles[:,1])))
        print('SMAPE3 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,2], pr_cobb_angles[:,2])))

        landmark_dist = np.asarray(landmark_dist, np.float32)
        print('2mm acc = {}'.format(np.sum(landmark_dist < 2.0) / max(len(landmark_dist), 1)))
        print('2.5mm acc = {}'.format(np.sum(landmark_dist < 2.5) / max(len(landmark_dist), 1)))
        print('3mm acc = {}'.format(np.sum(landmark_dist < 3.0) / max(len(landmark_dist), 1)))
        print('4mm acc = {}'.format(np.sum(landmark_dist < 4.0) / max(len(landmark_dist), 1)))
        print('mean landmark error (mm) is {}'.format(np.mean(landmark_dist)))

        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1./np.mean(total_time)))
        if dump_sample_csv:
            valid_errors = np.asarray([row[1] for row in per_sample_rows if np.isfinite(row[1])], dtype=np.float32)
            mean_error = float(np.mean(valid_errors)) if valid_errors.size else 1.0
            with open(dump_sample_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['img_id', 'mean_error_mm', 'sample_weight'])
                for img_id, sample_error in per_sample_rows:
                    if not np.isfinite(sample_error):
                        weight = 1.0
                    else:
                        weight = float(sample_error / (mean_error + 1e-6))
                    writer.writerow([img_id, sample_error, weight])
            print('saved per-sample errors to {}'.format(dump_sample_csv))


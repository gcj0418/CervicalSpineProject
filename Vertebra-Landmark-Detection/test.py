import os

import cv2
import decoder
import numpy as np
import torch

from dataset import BaseDataset
from models import spinal_net
import draw_points
import transform


class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {
            'hm': args.num_classes,
            'reg': 2 * args.num_classes,
            'wh': 2 * 4,
        }

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

    def test(self, args, save):
        save_path = args.weights_dir if getattr(args, 'weights_dir', None) else 'weights_' + args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        base_output_dir = getattr(args, 'output_dir', os.path.join('outputs', 'inference_vld'))
        hospital_name = getattr(args, 'hospital_name', 'RUIJIN')
        output_dir = os.path.join(base_output_dir, hospital_name, 'visualizations')
        compare_dir = os.path.join(base_output_dir, hospital_name, 'visualizations_compare')
        side_by_side_dir = os.path.join(base_output_dir, hospital_name, 'visualizations_side_by_side')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(compare_dir, exist_ok=True)
        os.makedirs(side_by_side_dir, exist_ok=True)

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio,
                               max_points=args.max_points)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)

        for cnt, data_dict in enumerate(data_loader):
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to(self.device)
            print('processing {}/{} image ... {}'.format(cnt, len(data_loader), img_id))
            with torch.no_grad():
                output = self.model(images)
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']

            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)
            pts0 = pts2.copy()
            pts0[:, :10] *= args.down_ratio

            print('total pts num is {}'.format(len(pts2)))

            ori_image = dsets.load_image(dsets.img_ids.index(img_id))
            gt_pts = dsets.load_gt_pts(dsets.load_annoFolder(img_id))
            ori_image_regress, gt_pts = transform.resize_with_letterbox(ori_image, gt_pts, (args.input_w, args.input_h), fill=0)
            ori_image_points = ori_image_regress.copy()
            ori_image_compare = ori_image_regress.copy()
            ori_image_side_by_side = ori_image_regress.copy()

            pts0 = np.asarray(pts0, np.float32)
            sort_ind = np.argsort(pts0[:, 1])
            pts0 = pts0[sort_ind]

            ori_image_regress, ori_image_points = draw_points.draw_landmarks_regress_test(
                pts0,
                ori_image_regress,
                ori_image_points,
            )
            ori_image_compare = draw_points.draw_landmarks_compare_test(
                gt_pts,
                pts0,
                ori_image_compare,
            )
            ori_image_side_by_side = draw_points.draw_landmarks_side_by_side_test(
                gt_pts,
                pts0,
                ori_image_side_by_side,
                ori_image_side_by_side,
            )

            cv2.imwrite(os.path.join(output_dir, f'{img_id}_regress.png'), ori_image_regress)
            cv2.imwrite(os.path.join(output_dir, f'{img_id}_points.png'), ori_image_points)
            cv2.imwrite(os.path.join(compare_dir, f'{img_id}_compare.png'), ori_image_compare)
            cv2.imwrite(os.path.join(side_by_side_dir, f'{img_id}_side_by_side.png'), ori_image_side_by_side)
            print('saved visualizations to {}'.format(output_dir))

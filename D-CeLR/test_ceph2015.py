#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time
import cv2
import numpy as np
import torch
from data.load_test_data2015 import TestData
from torch.utils.data import DataLoader
from utils import  decode_reg, cal_acc, cal_class, view_p
from net.ceph_reg_refine_net import get_model
import config.config as cfg



def model_initial(model, model_name):
    # 加载预训练模型
    pretrained_dict = torch.load(model_name)["model"]
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dictf = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dictf)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = "G:/ISBI_data/Test1Data/"
    label_path = "G:/ISBI_data/AnnotationsByMD/"
    test_loader = DataLoader(TestData(file_path, label_path), num_workers=0,
                             batch_size=1, shuffle=False, drop_last=False)

    # Try to load models
    num_layers =34
    head_conv = 256
    heads = {'hm': 1, 'class': cfg.PointNms}
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    model_name = "./save_model/bestrefine.pth"
    model =model_initial(model, model_name)

    model.cuda()
    model.eval()
    num = 0
    total_masks = 0
    total_counts = []
    diff_coords =[]
    gtclassv = []
    pdclassv = []
    for rowImg, test_data, label_coords_, scalek, sizek in test_loader:
        test_data = test_data.cuda().float()
        # with autocast():
        scalek = scalek.squeeze().numpy()
        sizek = sizek.squeeze().numpy()
        with torch.no_grad():
            outputs, inint_coords, hotmap = model(test_data)
            pred = outputs[-1][:, :, :2]

            key_points, mask_ = decode_reg(pred)
            pcoords = key_points[:, 1:3] * scalek
            offv = 300
            x1, y1 = int(np.min(pcoords[:,0]))-offv+100, int(np.min(pcoords[:,1]))-offv
            x2, y2 = int(np.max(pcoords[:,0]))+offv+100, int(np.max(pcoords[:,1]))+offv
            cropimg = torch.squeeze(rowImg)[y1:y2, x1:x2, :].numpy()
            lheight, lwidth, _ = cropimg.shape
            rowImg_= np.copy(cropimg)
            cropimg = cv2.resize(cropimg, (cfg.IMG_Width, cfg.IMG_Height), interpolation=cv2.INTER_LINEAR)
            cropimg = cropimg / np.max(cropimg)
            scalex = 1.0 * lwidth / cfg.IMG_Width
            scaley = 1.0 * lheight / cfg.IMG_Height
            scalek = np.array([[scalex, scaley]])
            cropimg = torch.tensor(cropimg).unsqueeze(dim=0).permute(0, 3, 1,2).cuda().float()


            outputs, inint_coords, hotmap = model(cropimg)
            pred = outputs[-1][:, :, :2]
            key_points, mask_ = decode_reg(pred)
            label_coords_ = label_coords_ - torch.tensor(np.array([[x1, y1]]))
            #
            label_coords_ = label_coords_.squeeze().numpy()
            # rowImg = rowImg.squeeze().numpy()

            counts, key_points = cal_acc(rowImg_, key_points, mask_, label_coords_, scalek)
            pclassv = cal_class(key_points)
            gclassv = cal_class(label_coords_)
            drawimg = view_p(rowImg_, key_points, mask_, label_coords_, scalek)
            DI = torch.squeeze(rowImg).numpy()
            DI[y1:y2, x1:x2, :] = drawimg
            total_counts.append(counts)
            total_masks = total_masks + np.sum(mask_)
            gtclassv.append(gclassv)
            pdclassv.append(pclassv)
            # if 19>np.sum(mask_):
            #     pred_numpy = pred_.astype(np.float32)
            #     dimg = view_p(rowImg, key_points, mask_, label_coords_, scalek)
            #     for si in range(pred_.shape[0]):
            #
            #         cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            #         cv2.imshow("img", pred_numpy[si,:,:])
            #         cv2.waitKey(0)
            # for si in range(pred_.shape[0]):
            # hotmap_numpy = hotmap.sigmoid().squeeze().detach().cpu().numpy().astype(np.float32)
            # cv2.imwrite("./outputs/2/" + str(num)+".jpg", DI)
            # cv2.namedWindow("dimg", cv2.WINDOW_NORMAL)
            # cv2.imshow("dimg", DI)
            # cv2.waitKey(0)

        num = num +1
        print(num)

    total_counts = np.array(total_counts)
    print(np.mean(total_counts))
    print(np.sum(total_counts< cfg.ERROR_RANGE[0], axis=0) / num)
    print(np.sum(total_counts< cfg.ERROR_RANGE[1], axis=0) / num)
    print(np.sum(total_counts< cfg.ERROR_RANGE[2], axis=0) / num)
    print(np.sum(total_counts< cfg.ERROR_RANGE[3], axis=0) / num)
    print(np.mean(total_counts, axis=0))
    total_points = num*cfg.PointNms
    print("total_points = ", total_points, "     total_masks = ", total_masks)
    print("2mm  acc = ", np.sum(total_counts< cfg.ERROR_RANGE[0])/total_points)
    print("2.5mm  acc = ", np.sum(total_counts< cfg.ERROR_RANGE[1])/total_points)
    print("3mm  acc = ", np.sum(total_counts< cfg.ERROR_RANGE[2])/total_points)
    print("4mm  acc = ", np.sum(total_counts< cfg.ERROR_RANGE[3])/total_points)
    gtclassv = np.array(gtclassv)
    pdclassv = np.array(pdclassv)
    maskv = (gtclassv ==pdclassv)
    print("classv = ", np.sum(maskv, axis=0) / num)

            # pred_numpy = torch.squeeze(pred).detach().cpu().numpy()
            #
            # for ci in range(pred_numpy.shape[0]):
            #     cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            #     cv2.imshow("img", pred_numpy[ci, :, :])
            #     cv2.waitKey(0)
    print("over")


if __name__ == "__main__":
    aa = np.array([0.88, 0.9,  0.75, 0.86, 0.84, 0.92, 0.88, 0.92])
    bb = np.array([0.74666667, 0.92, 0.75333333, 0.90666667, 0.82, 0.82, 0.87333333, 0.94])
    cc = (aa+bb)/2.0
    print(cc)
    print()
    test()


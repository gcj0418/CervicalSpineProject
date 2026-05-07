import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from config import config as cfg


def decode_nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def decode_key_points(pred_, heat, cls, th=0.15):

    h, w = heat.shape
    ks = cfg.IMG_Width / w
    cls = cls.squeeze().detach().cpu().numpy()
    yindex, xindex = np.where(heat > th)
    cindex = np.argmax(cls,axis=0)

    mask_ = np.zeros((cfg.PointNms), np.int32)
    key_points = np.zeros((cfg.PointNms, 4), np.float32)
    ksize = 5
    for ci in range(yindex.shape[0]):
        x, y = int(xindex[ci]), int(yindex[ci])
        class_index = cindex[y, x]
        x1, x2 = max(x-ksize, 0), min(x-ksize+1, w)
        y1, y2 = max(y-ksize, 0), min(y-ksize+1, h)
        scoreV = np.sum(pred_[y1: y2, x1: x2])
        if 1 != mask_[class_index]:
            mask_[class_index] = 1
            key_points[class_index] = np.array([class_index, x*ks, y*ks, scoreV])
        else:
            if scoreV > key_points[class_index, 3]:
                key_points[class_index] = np.array([class_index, x*ks, y*ks, scoreV])


    return key_points, mask_

def decode_reg(prd):
    # cenp = np.array([[(cfg.IMG_Width-1) / 2, (cfg.IMG_Height-1) / 2]])

    prd_numpy = torch.squeeze(prd).detach().cpu().numpy() * np.array([[cfg.IMG_Height, cfg.IMG_Width]])#cenp +cenp

    mask_ = np.zeros((cfg.PointNms), np.int32)
    key_points = np.zeros((cfg.PointNms, 4), np.float32)
    for i in range(prd_numpy.shape[0]):
        coord = prd_numpy[i]
        key_points[i] = np.array([i, coord[0], coord[1], 0])
        mask_[i] =1

    return key_points, mask_



def cal_acc(img, key_points, mask_, gcoords, scalek, resov=0.1):

    pcoords = key_points[:, 1:3] * scalek
    counts = np.zeros((cfg.PointNms), np.float32)
    resov_arr = np.asarray(resov, dtype=np.float32).reshape(-1)
    for pi in range(pcoords.shape[0]):
        if 1 == mask_[pi]:
            pcoord = pcoords[pi]
            gcoord = gcoords[pi]
            diff = pcoord - gcoord
            if resov_arr.size >= 2:
                value = np.sqrt((diff[0] * resov_arr[0]) ** 2 + (diff[1] * resov_arr[1]) ** 2)
            else:
                value = np.sqrt(np.sum(np.power(diff, 2))) * float(resov_arr[0] if resov_arr.size else 0.1)
            counts[pi] = value
    return counts, pcoords

def cal_class(key_points):
    ANB,SNB,SNA,FHI,FHA, MW, ODI, APDI = None,None,None,None,None,None,None,None

    a = key_points[5-1] - key_points[2-1]
    b = key_points[6-1] - key_points[2-1]
    aa= a.dot(b)
    bb = (np.sqrt(np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))))
    ANB = np.degrees(np.arccos(aa/bb))

    a = key_points[1-1] - key_points[2-1]
    b = key_points[6-1] - key_points[2-1]
    SNB = np.degrees(np.arccos(a.dot(b)/(np.sqrt(np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2)))) ))

    a = key_points[1-1] - key_points[2-1]
    b = key_points[5-1] - key_points[2-1]
    SNA = np.degrees(np.arccos(a.dot(b)/(np.sqrt(np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2)))) ))

    a = key_points[10-1] - key_points[1-1]
    b = key_points[8-1] - key_points[2-1]
    FHI = (np.sqrt(np.sum(np.power(a, 2)))/np.sqrt(np.sum(np.power(b, 2))))

    a = key_points[2 - 1] - key_points[1 - 1]
    b = key_points[9 - 1] - key_points[10 - 1]
    FHA = np.degrees(np.arccos(a.dot(b) / (np.sqrt(np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))))))

    a = key_points[12-1] - key_points[11-1]
    MW = np.sqrt(np.sum(np.power(a, 2)))*0.1
    if key_points[12-1][0] <= key_points[11-1][0]:
        MW = -np.sqrt(np.sum(np.power(a, 2)))*0.1

    a = key_points[5 - 1] - key_points[6 - 1]
    b = key_points[10 - 1] - key_points[8 - 1]
    ODI1 = np.degrees(np.arccos(a.dot(b) / (np.sqrt(np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))))))
    a = key_points[18 - 1] - key_points[17 - 1]
    b = key_points[3 - 1] - key_points[4 - 1]
    ODI2 = -np.degrees(np.arccos(a.dot(b) / (np.sqrt(np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))))))
    # a = a / (np.sqrt(np.sum(a, 2)))
    # b = b / (np.sqrt(np.sum(a, 2)))
    # if a[1] > b[1]:
    #     ODI2 = -ODI2
    ODI = ODI1 + ODI2

    a = key_points[4 - 1] - key_points[3 - 1]
    b = key_points[7 - 1] - key_points[2 - 1]
    APDI1 = np.degrees(np.arccos(a.dot(b) / (np.sqrt(np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))))))

    a = key_points[2 - 1] - key_points[7 - 1]
    b = key_points[5 - 1] - key_points[6 - 1]
    APDI2 = -np.degrees(np.arccos(a.dot(b) / (np.sqrt(np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))))))
    # a = a / (np.sqrt(np.sum(a, 2)))
    # b = b / (np.sqrt(np.sum(a, 2)))
    # if a[0] < b[0]:
    #     APDI2 = -APDI2

    a = key_points[3 - 1] - key_points[4 - 1]
    b = key_points[18 - 1] - key_points[17 - 1]
    APDI3 = -np.degrees(np.arccos(a.dot(b) / (np.sqrt(np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))))))

    APDI = APDI1 + APDI2 + APDI3

    typev = np.zeros((8), np.int32)
    if ANB >=3.2 and ANB <= 5.7:
        typev[0] = 1
    elif ANB > 5.7:
        typev[0] = 2
    else:
        typev[0] = 3
    if SNB >=74.6 and SNB <= 78.7:
        typev[1] = 1
    elif SNB < 74.6:
        typev[1] = 2
    else:
        typev[1] = 3

    if SNA >=79.4 and SNA <= 83.2:
        typev[2] = 1
    elif SNA > 83.2:
        typev[2] = 2
    else:
        typev[2] = 3

    if ODI >=68.43 and ODI <= 80.57:
        typev[3] = 1
    elif ODI > 80.57:
        typev[3] = 2
    else:
        typev[3] = 3

    if APDI >=77.6 and APDI <= 85.2:
        typev[4] = 1
    elif APDI < 77.6:
        typev[4] = 2
    else:
        typev[4] = 3

    if FHI >=0.65 and FHI <= 0.75:
        typev[5] = 1
    elif FHI > 0.75:
        typev[5] = 2
    else:
        typev[5] = 3

    if FHA >=26.8 and FHA <= 31.4:
        typev[6] = 1
    elif FHA > 31.4:
        typev[6] = 2
    else:
        typev[6] = 3

    if MW >0 and MW <= 4.5:
        typev[7] = 1
    elif MW <= 0:
        typev[7] = 2
    elif MW > 4.5:
        typev[7] = 3

    return typev



def view_p(img, key_points, mask_, gcoords, scalek):

    pcoords = key_points#[:, 1:3] * scalek
    img = img.astype(np.float32)
    for pi in range(pcoords.shape[0]):
        if 1== mask_[pi]:
           coord = pcoords[pi].astype(np.int32)
           img = cv2.drawMarker(img, (coord[0], coord[1]), (255, 0, 255), cv2.MARKER_DIAMOND, 8, thickness=16)

    for pi in range(gcoords.shape[0]):
        coord = gcoords[pi].astype(np.int32)
        img = cv2.drawMarker(img, (coord[0], coord[1]), (255, 0, 0), cv2.MARKER_DIAMOND, 8, thickness=16)

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    #
    # print("")
    return img


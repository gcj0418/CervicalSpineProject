#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time
import argparse
import torch
import random
import cv2
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data.load_train_data2023 import TrainData
from data.load_test_load2023 import TestData
from torch.utils.data import DataLoader
from utils import  decode_reg, cal_acc
from net.ceph_reg_refine_net import get_model
from net.reg_loss import rcal_loss
import config.config as cfg


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('./outputs/' + args.exp_name):
        os.makedirs('./outputs/' + args.exp_name)
    if not os.path.exists('./outputs/' + args.exp_name + '/' + 'models'):
        os.makedirs('./outputs/' + args.exp_name + '/' + 'models')
    os.system('cp main_cls.py outputs' + '/' + args.exp_name + '/' + 'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(file_dir, file_list, type_str)
        else:
            if file_.rfind(type_str) !=-1:
                file_list.append(path)

def train(args, io):
    file_list = []
    type_str = "m.npy"
    file_dir = "G:/ISBI2023/train_data/"
    get_files(file_dir, file_list, type_str)
    random.shuffle(file_list)


    file_ = open(file_dir + "train1.txt", "w")
    for i in range(0, 500, 1):
        file_.write(file_list[i] + '\n')
    file_.close()
    file_ = open(file_dir + "test1.txt", "w")
    for i in range(500, len(file_list), 1):
        file_.write(file_list[i] + '\n')
    file_.close()

    label_path = "G:/ISBI2023/train_data/"
    train_loader = DataLoader(TrainData(file_list[:500], label_path), num_workers=0,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    label_path = "G:/ISBI2023/train_data/"
    test_loader = DataLoader(TestData(file_list[500:], label_path), num_workers=0,
                             batch_size=1, shuffle=True, drop_last=False)


    num_layers =34
    model = get_model(num_layers=num_layers, NLayer1=2, NLayer2=4, numchannel=512)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if args.use_sgd:
        print("Use SGD")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=0.5e-6, last_epoch = -1)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)


    model.cuda()
    model.train()
    scaler = GradScaler()
    inter_nums = len(train_loader)
    total_acc = 0
    TMAXcounts = None
    for epoch in range(0, args.epochs):
        ####################
        # Train
        ####################

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_loss = 0.0
        inint_loss = 0.0
        loss_ml = 0.0
        loss_dl = 0.0
        edgeloss = 0
        # for data, edges, label in train_loader:
        tic = time.time()
        nums = 0
        model.train()
        tic = time.time()
        for train_data, hotmap, hot_mapl, offestxy, mask_, label_re in train_loader:
            train_data = train_data.cuda().float()
            hotmap = hotmap.cuda().float()
            hot_mapl = hot_mapl.cuda().float()
            offestxy = offestxy.cuda().float()
            mask_ = mask_.cuda().float()
            label_re = label_re.cuda().float()

            nums = nums +1
            opt.zero_grad()
            with autocast():

                outputs, inint_coords, prehotmap = model(train_data)
                inint_loss_ = model.Initloss(inint_coords, label_re)
                loss1 = model.loss(outputs[0], label_re)
                loss2 = model.loss(outputs[1], label_re)
                loss3 = model.loss(outputs[2], label_re)
                loss4 = model.loss(outputs[3], label_re)
                loss = loss1 + loss2 + loss3 + loss4

                loss_ml_, loss_dl_, loss_re_, loss_fl_ = rcal_loss(prehotmap, hot_mapl)
                edgeloss_ = loss_fl_  # edge_loss(outputs, label_re.repeat(1, 4, 1))

                loss = inint_loss_ + loss + loss_ml_ + loss_dl_  # + edgeloss_

                scaler.scale(loss).backward()
                # Unscales gradients and calls
                # or skips optimizer.step()
                scaler.step(opt)
                # Updates the scale for next iteration
                scaler.update()

            train_loss += loss.item()
            inint_loss += inint_loss_.item()
            loss_ml += loss_ml_.item()
            loss_dl += loss_dl_.item()
            edgeloss += edgeloss_.item()
            if nums % cfg.VIEW_NUMS == 0:
                toc = time.time()
                train_loss = train_loss/ (cfg.VIEW_NUMS)
                inint_loss = inint_loss/ (cfg.VIEW_NUMS)
                loss_ml = loss_ml/ (cfg.VIEW_NUMS)
                loss_dl = loss_dl/ (cfg.VIEW_NUMS)
                edgeloss = edgeloss/ (cfg.VIEW_NUMS)

                print("lr = ", opt.param_groups[0]['lr'], "loss1 = ", loss1.item(), "loss2 = ", loss2.item(),"loss3 = ", loss3.item(), "loss4 = ", loss4.item())
                outstr = 'epoch %d /%d,epoch %d /%d, loss: %.6f, inint_loss: %.6f, loss_ml: %.6f, loss_dl: %.6f, edgeloss: %.6f, const time: %.6f' % (
                 epoch,args.epochs, nums, inter_nums, train_loss, inint_loss, loss_ml, loss_dl, edgeloss, toc - tic)

                io.cprint(outstr)
                train_loss = 0.0
                inint_loss = 0.0
                loss_ml = 0.0
                loss_dl = 0.0
                edgeloss = 0.0
                tic = time.time()
        if 0 == epoch % 1 and epoch>=0:
            model.eval()
            num = 0
            total_masks = 0
            total_counts = []
            for rowImg, test_data, label_coords_, scalek, sizek, resov in test_loader:
                test_data = test_data.cuda().float()
                # with autocast():
                resov = resov.squeeze().numpy()
                scalek = scalek.squeeze().numpy()
                # label_coords_ = label_coords_.squeeze().numpy()
                with torch.no_grad():
                    outputs, inint_coords, hotmap = model(test_data)
                    pred = outputs[-1][:, :, :2].clip(0, 0.999)

                    key_points, mask_ = decode_reg(pred)
                    pcoords = key_points[:, 1:3] * scalek
                    offv = 300
                    x1, y1 = max(int(np.min(pcoords[:, 0])) - offv + 100, 0),  max(int(np.min(pcoords[:, 1])) - offv, 0)
                    x2, y2 = int(np.max(pcoords[:, 0])) + offv + 100, int(np.max(pcoords[:, 1])) + offv
                    cropimg = torch.squeeze(rowImg)[y1:y2, x1:x2, :].numpy()
                    lheight, lwidth, _ = cropimg.shape
                    rowImg_ = np.copy(cropimg)
                    cropimg = cv2.resize(cropimg, (cfg.IMG_Width, cfg.IMG_Height), interpolation=cv2.INTER_LINEAR)
                    cropimg = cropimg / np.max(cropimg)
                    scalex = 1.0 * lwidth / cfg.IMG_Width
                    scaley = 1.0 * lheight / cfg.IMG_Height
                    scalek = np.array([[scalex, scaley]])
                    cropimg = torch.tensor(cropimg).unsqueeze(dim=0).permute(0, 3, 1, 2).cuda().float()

                    outputs, inint_coords, hotmap = model(cropimg)
                    pred = outputs[-1][:, :, :2]
                    key_points, mask_ = decode_reg(pred)
                    label_coords_ = label_coords_ - torch.tensor(np.array([[x1, y1]]))
                    #
                    label_coords_ = label_coords_.squeeze().numpy()
                    counts, key_points = cal_acc(rowImg_, key_points, mask_, label_coords_, scalek, resov)
                    total_counts.append(counts)
                    total_masks = total_masks + np.sum(mask_)

                    num = num + 1

            total_counts = np.array(total_counts)
            TWOMM = np.sum(total_counts < cfg.ERROR_RANGE[0], axis=0)
            print(TWOMM / num)
            total_points = num * cfg.PointNms
            print("total_points = ", total_points, "     total_masks = ", total_masks)
            print("2mm  acc = ", np.sum(total_counts < cfg.ERROR_RANGE[0]) / total_points)
            print("2.5mm  acc = ", np.sum(total_counts < cfg.ERROR_RANGE[1]) / total_points)
            print("3mm  acc = ", np.sum(total_counts < cfg.ERROR_RANGE[2]) / total_points)
            print("4mm  acc = ", np.sum(total_counts < cfg.ERROR_RANGE[3]) / total_points)

            if total_acc < np.sum(total_counts < cfg.ERROR_RANGE[0]) / total_points:
                total_acc = np.sum(total_counts < cfg.ERROR_RANGE[0]) / total_points
                TMAXcounts = total_counts
                torch.save({'model': model.state_dict(), 'epoch': epoch},
                           'outputs/' + 'best.pth')
            print("total_acc = ", total_acc)
            print(np.mean(TMAXcounts))
            print(np.sum(TMAXcounts < cfg.ERROR_RANGE[0], axis=0) / num)
            print(np.sum(TMAXcounts < cfg.ERROR_RANGE[1], axis=0) / num)
            print(np.sum(TMAXcounts < cfg.ERROR_RANGE[2], axis=0) / num)
            print(np.sum(TMAXcounts < cfg.ERROR_RANGE[3], axis=0) / num)
            print(np.mean(TMAXcounts, axis=0))
            print("2mm  acc = ", np.sum(TMAXcounts < cfg.ERROR_RANGE[0]) / total_points)
            print("2.5mm  acc = ", np.sum(TMAXcounts < cfg.ERROR_RANGE[1]) / total_points)
            print("3mm  acc = ", np.sum(TMAXcounts < cfg.ERROR_RANGE[2]) / total_points)
            print("4mm  acc = ", np.sum(TMAXcounts < cfg.ERROR_RANGE[3]) / total_points)

        if (epoch) % cfg.SAVE_MODEL == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch}, 'outputs/ceph_model_' + str(epoch)+ '.pth')


if __name__ == "__main__":


    torch.backends.cudnn.enabled = True
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='cls_1024', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=5, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=501, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')#
    parser.add_argument('--lr', type=float, default=0.2*1e-3 , metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=2048, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)


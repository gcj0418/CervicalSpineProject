# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim


def create_logger(cfg, cfg_name, phase='train'):
    dataset = cfg.DATASET.DATASET
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    epoch_tag = 'e{}'.format(getattr(cfg.TRAIN, 'END_EPOCH', '')) if hasattr(cfg, 'TRAIN') else ''
    phase_prefix = 'training' if phase == 'train' else 'inference'
    final_output_dir = Path(cfg.OUTPUT_DIR) / f'{phase_prefix}_{dataset.lower()}_{cfg_name}_{epoch_tag}'.rstrip('_')

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = final_output_dir / 'logs'
    checkpoints_dir = final_output_dir / 'checkpoints'
    predictions_dir = final_output_dir / 'predictions'
    visualizations_dir = final_output_dir / 'visualizations'
    for directory in [logs_dir, checkpoints_dir, predictions_dir, visualizations_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = logs_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = logs_dir / 'tensorboard' / (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def save_checkpoint(states, predictions, is_best,
                    output_dir, filename='checkpoint.pth'):
    preds = predictions.cpu().data.numpy()
    torch.save(states, os.path.join(output_dir, filename))
    torch.save(preds, os.path.join(output_dir, 'current_pred.pth'))

    latest_path = os.path.join(output_dir, 'latest.pth')
    shutil.copy2(os.path.join(output_dir, filename), latest_path)

    if is_best and 'state_dict' in states.keys():
        torch.save(states['state_dict'].module, os.path.join(output_dir, 'model_best.pth'))


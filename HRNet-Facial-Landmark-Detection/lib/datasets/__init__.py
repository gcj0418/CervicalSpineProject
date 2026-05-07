# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

try:
    from .aflw import AFLW
except Exception:
    AFLW = None

try:
    from .cofw import COFW
except Exception:
    COFW = None

try:
    from .face300w import Face300W
except Exception:
    Face300W = None

from .spine_npy import SpineNpy

try:
    from .wflw import WFLW
except Exception:
    WFLW = None

__all__ = ['AFLW', 'COFW', 'Face300W', 'SpineNpy', 'WFLW', 'get_dataset']


def get_dataset(config):

    if config.DATASET.DATASET == 'AFLW':
        return AFLW
    elif config.DATASET.DATASET == 'COFW':
        return COFW
    elif config.DATASET.DATASET == '300W':
        return Face300W
    elif config.DATASET.DATASET == 'WFLW':
        return WFLW
    elif config.DATASET.DATASET in ('RENJI', 'RUIJIN'):
        return SpineNpy
    else:
        raise NotImplemented()


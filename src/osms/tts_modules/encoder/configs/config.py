from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN
import os


def get_default_encoder_config():
    _C = CN()
    _C.VERBOSE = True

    _C.AUDIO = CN()
    _C.AUDIO.MEL_N_CHANNELS = 40
    # TODO: add all others
    # ...

    _C.MODEL = CN()
    _C.MODEL.PRETRAINED = True
    _C.MODEL.CHECKPOINT_DIR_PATH = os.path.join("checkpoints", "encoder")
    # TODO: add all others
    # ...

    _C.freeze()
    return _C


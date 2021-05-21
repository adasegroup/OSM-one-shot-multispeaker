import os
from yacs.config import CfgNode as CN
from osms.tts_modules.synthesizer.configs import add_default_signal_processing_config


def get_default_vocoder_config():
    _C = CN()
    _C.VERBOSE = True

    # Add Signal Processing main_configs (used in both synthesizer and vocoder)
    _C = _add_default_signal_processing_config(_C, freeze=False)

    # Add WaveRNN model main_configs
    _C = _add_default_wavernn_config(_C, freeze=False)

    _C.freeze()
    return _C


def _add_default_wavernn_config(config, freeze=True):
    if config.is_frozen():
        config.defrost()

    config.MODEL = CN()
    config.MODEL.PRETRAINED = True
    config.MODEL.CHECKPOINT_DIR_PATH = os.path.join("checkpoints", "vocoder")
    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
    config.MODEL.MODE = "RAW"
    # NB - this needs to correctly factorise hop_length
    config.MODEL.UPSAMPLE_FACTORS = (5, 5, 8)
    config.MODEL.RNN_DIMS = 512
    config.MODEL.FC_DIMS = 512
    config.MODEL.COMPUTE_DIMS = 128
    config.MODEL.RES_OUT_DIMS = 128
    config.MODEL.RES_BLOCKS = 10

    # Training
    config.MODEL.BATCH_SIZE = 100
    config.MODEL.LR = 1e-4
    # number of samples to generate at each checkpoint
    config.MODEL.GEN_AT_CHECKPOINT = 5
    # this will pad the input so that the resnet can 'see' wider than input length
    config.MODEL.PAD = 2
    config.MODEL.SEQ_LEN = config.SP.HOP_SIZE * 5  # must be a multiple of hop_length

    # Generating / Synthesizing
    # very fast (realtime+) single utterance batched generation
    config.MODEL.GEN_BATCHED = True
    # target number of samples to be generated in each batch entry
    config.MODEL.TARGET = 8000
    config.MODEL.OVERLAP = 400

    if freeze:
        config.freeze()
    return config


def _add_default_signal_processing_config(config, freeze=True):
    if config.is_frozen():
        config.defrost()

    config = add_default_signal_processing_config(config, freeze=freeze)
    config.SP.BITS = 9
    config.SP.MU_LAW = True

    if freeze:
        config.freeze()
    return config

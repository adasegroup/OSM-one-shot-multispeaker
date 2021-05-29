import os
from yacs.config import CfgNode as CN
from osms.tts_modules.synthesizer.configs import add_default_signal_processing_config


def get_default_vocoder_config():
    """
    Creates the configuration CfgNode object for Vocoder and fills it with default values

    :return: Instance of CfgNode
    """
    _C = CN()
    _C.VERBOSE = True

    # Add Signal Processing main_configs (used in both synthesizer and vocoder)
    _C = _add_default_signal_processing_config(_C, freeze=False)

    # Add WaveRNN model main_configs
    _C = _add_default_wavernn_config(_C, freeze=False)

    _C.DATA = CN()
    _C.DATA.DATASET_ROOT_PATH = ""
    _C.DATA.VOC_DIR = os.path.join(_C.DATA.DATASET_ROOT_PATH, "SV2TTS", "vocoder")

    _C.freeze()
    return _C


def _add_default_wavernn_config(config, freeze=True):
    """
    Adds basic configurations for WaveRNN

    :param config: Synthesizer config
    :param freeze: Flag defines whether to freeze the configs after updating or nor
    :return: updated config
    """

    if config.is_frozen():
        config.defrost()

    config.MODEL = CN()
    config.MODEL.PRETRAINED = True
    config.MODEL.CHECKPOINT_DIR_PATH = "checkpoints"
    config.MODEL.CHECKPOINT_NAME = "test"
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
    config.MODEL.SAVE_EVERY = 5
    config.MODEL.GROUND_TRUTH = True
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
    """
   Adds SP CfgNode to :param config: containing signal processing parameters.
   Signal Processing parameters are used in both synthesizer and vocoder.
   Here most part of attributes are loaded from synthesizer's config.

   :param config: Vocoder config
   :param freeze: Flag defines whether to freeze :param config: after adding SP CfgNode
   :return: updated config
   """

    if config.is_frozen():
        config.defrost()

    config = add_default_signal_processing_config(config, freeze=freeze)
    config.SP.BITS = 9
    config.SP.MU_LAW = True

    if freeze:
        config.freeze()
    return config

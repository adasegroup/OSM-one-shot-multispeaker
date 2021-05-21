from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from yacs.config import CfgNode as CN


def get_default_synthesizer_config():
    _C = CN()
    _C.VERBOSE = True
    _C.PRETRAINED = True
    _C.CHECKPOINT_DIR_PATH = os.path.join("checkpoints", "synthesizer")

    # Signal Processing (used in both synthesizer and vocoder)
    add_default_signal_processing_config(_C, freeze=False)

    # Tacotron Text-to-Speech (TTS)
    _C.TTS = CN()
    _C.TTS.EMBED_DIMS = 512  # Embedding dimension for the graphemes/phoneme inputs
    _C.TTS.ENCODER_DIMS = 256
    _C.TTS.DECODER_DIMS = 128
    _C.TTS.POSTNET_DIMS = 512
    _C.TTS.ENCODER_K = 5
    _C.TTS.LSTM_DIMS = 1024
    _C.TTS.POSTNET_K = 5
    _C.TTS.NUM_HIGHWAYS = 4
    _C.TTS.DROPOUT = 0.5
    _C.TTS.CLEANER_NAMES = ["english_cleaners"]
    _C.TTS.STOP_THRESHOLD = -3.4

    # Tacotron Training
    # Progressive training schedule: (r, lr, step, batch_size),
    # where r = reduction factor (# of mel frames synthesized for each decoder iteration),
    # lr = learning rate
    _C.TTS.SCHEDULE = [(2, 1e-3, 20_000, 12),
                       (2, 5e-4, 40_000, 12),
                       (2, 2e-4, 80_000, 12),
                       (2, 1e-4, 160_000, 12),
                       (2, 3e-5, 320_000, 12),
                       (2, 1e-5, 640_000, 12)
                       ]
    # Clips the gradient norm to prevent explosion
    # Set to None if not needed
    _C.TTS.CLIP_GRAD_NORM = 1.0
    # Number of steps between model evaluation (sample generation)
    # Set to -1 to generate after completing epoch, or 0 to disable
    _C.TTS.EVAL_INTERVAL = 500
    _C.TTS.EVAL_NUM_SAMPLES = 1


    _C.freeze()
    return _C


def add_default_signal_processing_config(config, freeze=True):
    """
    Adds SP CfgNode to :param config: containing signal processing parameters.
    Signal Processing parameters are used in both synthesizer and vocoder.
    :param config: Synthesizer or Vocoder config
    :param freeze: Flag defines whether to freeze :param config: after adding SP CfgNode
    :return: updated config
    """
    if config.is_frozen():
        config.defrost()
    config.SP = CN()

    config.SP.SAMPLE_RATE = 16000
    config.SP.N_FFT = 800
    config.SP.NUM_MELS = 80
    config.SP.HOP_SIZE = 200
    config.SP.WIN_SIZE = 800
    config.SP.FMIN = 55
    config.SP.MIN_LEVEL_DB = -100
    config.SP.REF_LEVEL_DB = 20
    config.SP.MAX_ABS_VALUE = 4.
    config.SP.PREEMPHASIS = 0.97
    config.SP.PREEMPHASIZE = True
    if freeze:
        config.freeze()
    return config

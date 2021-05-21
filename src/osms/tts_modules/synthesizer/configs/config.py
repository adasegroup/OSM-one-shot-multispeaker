import os
from yacs.config import CfgNode as CN


def get_default_synthesizer_config():
    _C = CN()
    _C.VERBOSE = True

    # Signal Processing (used in both synthesizer and vocoder)
    _C = add_default_signal_processing_config(_C, freeze=False)

    # Tacotron Text-to-Speech (TTS)
    _C = _add_default_tacotron_tts_configs(_C, freeze=False)

    # Data Preprocessing
    _C = _add_default_data_processing_configs(_C, freeze=False)

    # Mel Visualization and Griffin-Lim
    _C = _add_default_mel_configs(_C, freeze=False)

    # Audio processing options
    _C = _add_default_audio_configs(_C, freeze=False)

    # SV2TTS
    _C = _add_default_sv2tts_configs(_C, freeze=False)

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


def _add_default_tacotron_tts_configs(config, freeze=True):
    if config.is_frozen():
        config.defrost()

    config.MODEL = CN()
    config.MODEL.PRETRAINED = True
    config.MODEL.CHECKPOINT_DIR_PATH = os.path.join("checkpoints", "synthesizer")
    config.MODEL.EMBED_DIMS = 512  # Embedding dimension for the graphemes/phoneme inputs
    config.MODEL.ENCODER_DIMS = 256
    config.MODEL.DECODER_DIMS = 128
    config.MODEL.POSTNET_DIMS = 512
    config.MODEL.ENCODER_K = 5
    config.MODEL.LSTM_DIMS = 1024
    config.MODEL.POSTNET_K = 5
    config.MODEL.NUM_HIGHWAYS = 4
    config.MODEL.DROPOUT = 0.5
    config.MODEL.CLEANER_NAMES = ["english_cleaners"]
    config.MODEL.STOP_THRESHOLD = -3.4

    # Tacotron Training
    # Progressive training schedule: (r, lr, step, batch_size),
    # where r = reduction factor (# of mel frames synthesized for each decoder iteration),
    # lr = learning rate
    config.MODEL.SCHEDULE = [(2, 1e-3, 20_000, 12),
                       (2, 5e-4, 40_000, 12),
                       (2, 2e-4, 80_000, 12),
                       (2, 1e-4, 160_000, 12),
                       (2, 3e-5, 320_000, 12),
                       (2, 1e-5, 640_000, 12)
                       ]
    # Clips the gradient norm to prevent explosion
    # Set to None if not needed
    config.MODEL.CLIP_GRAD_NORM = 1.0
    # Number of steps between model evaluation (sample generation)
    # Set to -1 to generate after completing epoch, or 0 to disable
    config.MODEL.EVAL_INTERVAL = 500
    config.MODEL.EVAL_NUM_SAMPLES = 1

    if freeze:
        config.freeze()
    return config


def _add_default_data_processing_configs(config, freeze=True):
    if config.is_frozen():
        config.defrost()

    config.DATA = CN()
    config.DATA.MAX_MEL_FRAMES = 900
    config.DATA.RESCALE = True
    config.DATA.RESCALING_MAX = 0.9
    config.DATA.SYNTHESIS_BATCH_SIZE = 16

    if freeze:
        config.freeze()
    return config


def _add_default_mel_configs(config, freeze=True):
    if config.is_frozen():
        config.defrost()

    config.MEL = CN()
    config.MEL.SIGNAL_NORMALIZATION = True
    config.MEL.POWER = 1.5
    config.MEL.GRIFFIN_LIM_ITERS = 60

    if freeze:
        config.freeze()
    return config


def _add_default_audio_configs(config, freeze=True):
    if config.is_frozen():
        config.defrost()

    config.AUDIO = CN()
    config.AUDIO.FMAX = 7600  # Should not exceed (sample_rate // 2)
    config.AUDIO.ALLOW_CLIPPING_IN_NORMALIZATION = True
    config.AUDIO.CLIP_MELS_LENGTH = True
    config.AUDIO.USE_LWS = False
    config.AUDIO.SYMMETRIC_MELS = True
    config.AUDIO.TRIM_SILENCE = True

    if freeze:
        config.freeze()
    return config


def _add_default_sv2tts_configs(config, freeze=True):
    if config.is_frozen():
        config.defrost()

    config.SV2TTS = CN()
    # Dimension for the speaker embedding
    config.SV2TTS.SPEAKER_EMBEDDING_SIZE = 256
    # Duration in seconds of a silence for an utterance to be split
    config.SV2TTS.SILENCE_MIN_DURATION_SPLIT = 0.4
    config.SV2TTS.UTTERANCE_MIN_DURATION = 1.6

    if freeze:
        config.freeze()
    return config

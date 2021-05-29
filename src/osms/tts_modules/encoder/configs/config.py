from yacs.config import CfgNode as CN
import os


def get_default_encoder_config():
    """
    Creates the configuration CfgNode object for Speaker Encoder and fills it with default values

    :return: Instance of CfgNode
    """
    _C = CN()
    _C.VERBOSE = False
    _C.DEVICE = 'cpu'

    _C.AUDIO = CN()
    _C.AUDIO.MEL_N_CHANNELS = 40
    _C.AUDIO.MEL_WINDOW_STEP = 10
    _C.AUDIO.MEL_WINDOW_LENGTH = 40
    _C.AUDIO.SAMPLING_RATE = 16000
    _C.AUDIO.PARTIAL_N_FRAMES = 160
    _C.AUDIO.INFERENCE_N_FRAMES = 80
    _C.AUDIO.VAD_WINDOW_LENGTH = 30
    _C.AUDIO.VAD_MOVING_AVERAGE_WIDTH = 8
    _C.AUDIO.VAD_MAX_SILENCE_LENGTH = 6
    _C.AUDIO.AUDIO_NORM_TARGET_dBFS = -30

    _C.MODEL = CN()
    _C.MODEL.PRETRAINED = True
    # _C.MODEL.CHECKPOINT_DIR_PATH = os.path.join("checkpoints", "encoder", "checkpoints")
    _C.MODEL.CHECKPOINT_DIR_PATH = "checkpoints"
    _C.MODEL.MODEL_HIDDEN_SIZE = 256
    _C.MODEL.MODEL_EMBEDDING_SIZE = 256
    _C.MODEL.MODEL_NUM_LAYERS = 3

    _C.DATASET = CN()
    _C.DATASET.ROOT = os.path.join("dataset", "LibriSpeech", "train-clean-100")
    _C.DATASET.OUTPUT_DIR = os.path.join("dataset", "LibriSpeech", "output")
    _C.DATASET.EXTENSION = "flac"

    _C.TRAIN = CN()
    _C.TRAIN.LEARNING_RATE_INIT = 1e-4
    _C.TRAIN.SPEAKERS_PER_BATCH = 64
    _C.TRAIN.UTTERANCES_PER_SPEAKER = 10
    _C.TRAIN.NUMBER_STEPS = 1000
    _C.TRAIN.CHECKPOINT_NAME = None
    _C.TRAIN.SAVE_N_STEPS = 5
    _C.TRAIN.OUT_DIR = "train_output"
    _C.TRAIN.RUN_ID = "EXP_1"

    _C.VALIDATE = CN()
    _C.VALIDATE.SPEAKERS_PER_BATCH = 64
    _C.VALIDATE.UTTERANCES_PER_SPEAKER = 10

    _C.freeze()
    return _C

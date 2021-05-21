from yacs.config import CfgNode as CN
import os


def update_config(config, args=None, update_file=None, update_list=None):
    config.defrost()

    if args is not None:
        if args.config_file_path is not None:
            config.merge_from_file(args.config_file_path)
        if args.opts is not None:
            config.merge_from_list(args.opts)
    if update_file is not None:
        config.merge_from_file(update_file)
    if update_list is not None:
        config.merge_from_list(update_list)

    config.freeze()
    return config


def add_cfg_node(config, node_name):
    config.defrost()
    setattr(config, node_name, CN())
    config.freeze()
    return config


def add_attribute(config, attr_name, attr_value):
    config.defrost()
    setattr(config, attr_name, attr_value)
    config.freeze()
    return config


def get_default_main_configs():
    _C = CN()

    # _C.SPEAKER_ENCODER_CONFIG_FILE = os.path.join("main_configs", "encoder")
    _C.SPEAKER_ENCODER_CONFIG_FILE = None
    # _C.SPEAKER_SYNTHESIZER_CONFIG_FILE = os.path.join("main_configs", "synthesizer")
    _C.SYNTHESIZER_CONFIG_FILE = None
    # _C.SPEAKER_VOCODER_CONFIG_FILE = os.path.join("main_configs", "vocoder")
    _C.VOCODER_CONFIG_FILE = None

    _C.SPEAKER_SPEECH_PATH = os.path.join("../audio_samples", "google_test.wav")
    _C.INPUT_TEXTS_PATH = os.path.join("../texts", "test1.txt")
    _C.OUTPUT_AUDIO_DIR = os.path.join("../result_speech")

    _C.freeze()
    return _C

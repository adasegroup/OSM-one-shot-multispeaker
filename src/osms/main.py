import argparse
import yaml
import os
from . import MultispeakerManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("main_config_path", type=str,
                        default='tts_modules/common/configs/main_config.yaml',
                        help='Path to main yaml configs')
    args = parser.parse_args()

    with open(args.main_config_path, "r") as ymlfile:
        main_configs = yaml.load(ymlfile)

    if not os.path.exists(main_configs["SPEAKER_SPEECH_PATH"]):
        os.makedirs(main_configs["SPEAKER_SPEECH_PATH"])

    if not os.path.exists(main_configs["INPUT_TEXTS_PATH"]):
        os.makedirs(main_configs["INPUT_TEXTS_PATH"])

    if not os.path.exists(main_configs["OUTPUT_AUDIO_DIR"]):
        os.makedirs(main_configs["OUTPUT_AUDIO_DIR"])

    multispeaker_manager = MultispeakerManager(main_configs)
    multispeaker_manager.inference()

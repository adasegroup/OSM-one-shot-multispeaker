import argparse
import yaml
from tts_modules.common.multispeaker import MultispeakerManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("main_config_path", type=str,
                        default='tts_modules/common/configs/main_config.yaml',
                        help='Path to main yaml configs')
    args = parser.parse_args()

    with open(args.main_config_path, "r") as ymlfile:
        main_configs = yaml.load(ymlfile)

    multispeaker_manager = MultispeakerManager(main_configs)
    multispeaker_manager.inference()

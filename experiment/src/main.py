import os

from kobe2 import kobe
from . import utils

logger = utils.get_logger(level='INFO')

def gather_configs(dir):
    logger.info(f"Gathering configurations from directory: {dir}")
    configs = []

    for file_name in os.listdir(dir):
        if file_name.endswith(".yaml"):
            file_path = os.path.join(dir, file_name)
            logger.info(f"Loading YAML file: {file_path}")

            configs.append(utils.load_yaml(file_path))

    return configs

def main():
    configs = gather_configs(dir='configs')

    logger.info("Starting KOBE...")
    kobe(configs, dir='static')

if __name__ == "__main__":
    main()
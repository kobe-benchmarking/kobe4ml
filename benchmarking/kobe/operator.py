import os
from . import utils

def gather_configs(dir):
    configs = []

    for file_name in os.listdir(dir):
        if file_name.endswith(".yaml"):
            file_path = os.path.join(dir, file_name)
            configs.append(utils.load_yaml(file_path))

    return configs

def run(input_dir, output_dir):
    experiments = gather_configs(input_dir)

    for exp in experiments:
        .....

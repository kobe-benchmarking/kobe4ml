import os
import sys
import importlib

from kobe2 import main as kobe
from . import utils

logger = utils.get_logger(level='INFO')

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

def gather_configs(dir):
    configs = []

    for file_name in os.listdir(dir):
        if file_name.endswith(".yaml"):
            file_path = os.path.join(dir, file_name)
            configs.append(utils.load_yaml(file_path))

    return configs

def create_exp_dir(config, dir):
    metadata = config['metadata']
    name = metadata['name']
    date = metadata['date'].replace('/', '_')

    exp_dir_name = f"{name}_{date}"
    exp_dir = os.path.join(dir, exp_dir_name)

    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Created directory: {exp_dir}")

    return exp_dir

def load_module(experiment, method):
    model_name = experiment['model']['name']
    model_url = experiment['model']['url']
    ds_url = experiment['process']['dataset']

    model_params = experiment['model']['parameters']
    process_params = experiment['process']['parameters']

    params = {'name': model_name, 'pth': model_url, 'ds': ds_url}
    params.update(model_params)
    params.update(process_params)

    module_name = f"algorithms.{model_params['id']}"
    module = importlib.import_module(module_name)

    return module, params

def main():
    experiments = gather_configs(dir='configs')
    calls = []

    for exp in experiments:
        exp_dir = create_exp_dir(config=exp, dir='experiments')

        process = exp['process']['name']
        method = "test" if process == 'inference' else "train"
        
        module, params = load_module(experiment=exp, method=method)

        call = lambda: getattr(module, method)(params)
        calls.append(call)

    kobe(calls, exp_dir)

if __name__ == "__main__":
    main()
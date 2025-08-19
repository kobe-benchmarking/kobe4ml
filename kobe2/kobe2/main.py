import os
import importlib
import pandas as pd
import sys
import subprocess
import tempfile
from urllib.parse import urlparse

from . import utils

logger = utils.get_logger(level='INFO')

def gather_configs(dir):
    """
    Gather all YAML configuration files from the specified directory.
    
    :param dir: Directory containing YAML configuration files.
    :return: List of configuration dictionaries loaded from YAML files.
    """
    logger.info(f"Gathering configurations from directory: {dir}")
    configs = []

    for filename in os.listdir(dir):
        if filename.endswith(".yaml"):
            file_path = utils.get_path(dir, filename=filename)
            logger.info(f"Loading YAML file: {file_path}.")

            configs.append(utils.load_yaml(path=file_path))

    return configs

def load_module(cfg):
    """
    Install and import a Python module directly in the active environment.

    cfg should be a dict with:
        - package: name of the package (required)
        - index_url: extra index URL (optional)
    """
    package = cfg["package"]
    index_url = cfg.get("index_url")

    try:
        module = importlib.import_module(package)
        logger.info(f"{package} already installed, using it.")
        return module
    except ImportError:
        logger.info(f"{package} not found, installing...")

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]
    if index_url:
        cmd += ["--extra-index-url", index_url]

    subprocess.check_call(cmd)

    module = importlib.import_module(package)
    logger.info(f"{package} loaded successfully.")
    return module

def load_impl_params(step):
    """
    Load parameters that configure the implementation for a specific step.

    :param step: Dictionary containing step information.
    :return: Dictionary of parameters.
    """
    logger.info(f"Loading parameters for step {step['id']}.")

    params = step['parameters']
    data = step['data']
    metrics = step['metrics']

    loader_module = load_module(cfg=data['loader'])

    ds_loc = data['location']
    ds_name = data['name']
    data_params = data['parameters']

    root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    ds_dir = utils.get_dir(root, ds_loc)
    
    loader_params = {'dir': ds_dir, 'name': ds_name}
    loader_params.update(data_params)

    dls = loader_module.preprocess(**loader_params)

    model_path = os.path.join(params['model_location'], params['model_name'])

    model_params = params['model'] if 'model' in params else {}
    process_params = params['process'] if 'process' in params else {}

    impl_params = {'pth': model_path, 'dls': dls}
    impl_params.update(model_params)
    impl_params.update(process_params)
    impl_params["metrics"] = metrics

    logger.info(f"Parameters for step {step['id']} loaded successfully.")

    return impl_params

def main(configs, dir='static'):
    """
    Run benchmarking experiments based on the provided configurations.
    
    :param configs: List of configuration dictionaries.
    :param dir: Directory to save experiment results.
    """
    logger.info("Starting KOBE benchmarking experiments...")
    
    experiments_data = {}
    methods_dict = {"prepare": "train", "work": "test"}

    for cfg in configs:
        metadata = cfg["metadata"]
        
        parent_id = metadata['parent_id']
        id = metadata['id']
        name = metadata['name']
        steps = cfg['steps']

        logger.info(f"Reading configuration {id} for {name}.")

        exp_dir = utils.get_dir(dir, parent_id)

        if parent_id not in experiments_data:
            experiments_data[parent_id] = {
                "calls": [],
                "steps": [],
                "results": []
            }

        for step in steps:
            process = step['type']
            method = methods_dict[process]

            logger.info(f"Processing step {step['id']} for {method}ing benchmarking.")

            impl = load_module(cfg=cfg['implementation'])
            params = load_impl_params(step)

            call = lambda impl=impl, m=method, p=params: getattr(impl, m)(p)

            experiments_data[parent_id]["calls"].append(call)
            experiments_data[parent_id]["steps"].append(step['id'])

    for parent_id, data in experiments_data.items():
        for i, call in enumerate(data["calls"]):
            metrics = call()
            data["results"].append(metrics)

            logger.info(f"Metrics for {data['steps'][i]}: {metrics}.")

    for parent_id, data in experiments_data.items():
        if data["results"]:
            df = pd.DataFrame(data["results"])

            exp_dir = os.path.join(dir, parent_id)
            csv_path = os.path.join(exp_dir, "results.csv")

            df.to_csv(csv_path, float_format="%.3f", index=False)
            logger.info(f"Results saved to {csv_path}.")

        else:
            logger.info("No results to save.")

    logger.info("よくやった!")
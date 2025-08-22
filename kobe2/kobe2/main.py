import os
import importlib
import pandas as pd
import sys
import subprocess
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

    # try:
    #     module = importlib.import_module(package)
    #     logger.info(f"{package} already installed, using it.")
    #     return module
    # except ImportError:
    #     logger.info(f"{package} not found, installing...")

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]
    if index_url:
        cmd += ["--extra-index-url", index_url]

    subprocess.check_call(cmd)

    module = importlib.import_module(package)
    logger.info(f"{package} loaded successfully.")

    return module

def prepare_dls(data, root_dir=None):
    """
    Prepare data loaders based on the provided data configuration.

    :param data: Dictionary containing data configuration.
    :param root_dir: Root directory for data storage.
    :return: Data loaders.
    """
    loader = data['loader']
    loc = data['location']
    name = data['name']
    params = data['parameters']

    ds_dir = utils.get_dir(root_dir, loc)

    loader_params = {
        "dir": ds_dir,
        "name": name,
        **params
    }

    loader_module = load_module(cfg=loader)
    dls = loader_module.preprocess(**loader_params)

    return dls

def resolve_path(root_dir, path):
    """
    Resolve a relative path to an absolute path based on the root directory.

    :param root_dir: Root directory.
    :param path: Relative path to resolve.
    :return: Absolute path or None if the input path is None.
    """
    if not path:
        return None
    dirs, filename = utils.split_path(path)
    full_path = utils.get_path(root_dir, dirs, filename=filename)

    return full_path

def load_params(step):
    """
    Load parameters that configure the implementation for a specific step. Handles optional keys: 'model', 'process', 'save_url', 'model_url'.

    :param step: Dictionary containing step information.
    :return: Dictionary of parameters.
    """
    root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

    step_id = step["id"]
    params = step['parameters']
    data = step['data']
    metrics = step['metrics']

    model_params = params.get('model')
    process_params = params.get('process')
    save_url = params.get('save_url')
    model_url = params.get('model_url')

    logger.info(f"Loading parameters for step {step_id}.")

    dls = prepare_dls(data, root_dir)
    logger.info(f"Data loaders prepared for step {step_id}.")

    save_path = resolve_path(root_dir, save_url)
    model_path = resolve_path(root_dir, model_url)

    impl_params = {
        "model_params": model_params if model_params else {},
        "model_url": model_path if model_path else {},
        "dls": dls,
        "metrics": metrics,
        "process_params": process_params if process_params else {},
        "save_url": save_path if save_path else {}
    }

    step_impl_params = {k: v for k, v in impl_params.items() if v}

    logger.info(f"Parameters for step {step_id} loaded successfully.")

    return step_impl_params

def main(configs, dir='static'):
    """
    Run benchmarking experiments based on the provided configurations.
    
    :param configs: List of configuration dictionaries.
    :param dir: Directory to save experiment results.
    """
    logger.info("Starting KOBE benchmarking experiments...")
    
    exp_data = {}
    methods_dict = {"prepare": "train", "work": "infer"}

    for cfg in configs:
        metadata = cfg["metadata"]
        
        parent_id = metadata['parent_id']
        id = metadata['id']
        name = metadata['name']
        steps = cfg['steps']

        logger.info(f"Reading configuration {id} for {name}.")

        exp_dir = utils.get_dir(dir, parent_id)

        if parent_id not in exp_data:
            exp_data[parent_id] = {
                "steps": [],
                "results": []
            }
        
        exp_data_parent = exp_data[parent_id]

        for step in steps:
            step_id = step['id']
            process = step['type']
            method = methods_dict[process]

            logger.info(f"Processing step {step_id} for {method}ing benchmarking.")

            impl = load_module(cfg=cfg['implementation'])
            params = load_params(step)

            call = lambda impl=impl, m=method, p=params: getattr(impl, m)(p)
            metrics = call()

            exp_data_parent["results"].append(metrics)
            exp_data_parent["steps"].append(step_id)

            logger.info(f"Metrics for step {step_id}: {metrics}.")

    for parent_id, data in exp_data.items():
        if data["results"]:
            df = pd.DataFrame(data["results"])

            exp_dir = os.path.join(dir, parent_id)
            csv_path = os.path.join(exp_dir, "results.csv")

            df.to_csv(csv_path, float_format="%.3f", index=False)
            logger.info(f"Results saved to {csv_path}.")

        else:
            logger.info("No results to save.")

    logger.info("よくやった!")
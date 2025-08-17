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

def load_module(module_url, target_dir=None):
    """
    Download & install a Python module from an explicit URL, then import it.
    
    :param module_url: Full URL to the wheel/tar.gz (from YAML).
    :param target_dir: Where to install the package (defaults to temp dir).
    :return: Imported Python module.
    """
    if target_dir is None:
        target_dir = os.path.join(tempfile.gettempdir(), "remote_modules")
    os.makedirs(target_dir, exist_ok=True)

    filename = os.path.basename(urlparse(module_url).path)
    module_name = filename.split("-")[0]

    try:
        return importlib.import_module(module_name)
    except ImportError:
        logger.info(f"{module_name} not found locally. Installing from {module_url}...")

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "--target", target_dir, module_url
    ])

    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    module = importlib.import_module(module_name)
    logger.info(f"Module {module_name} loaded successfully from {module_url}.")

    return module

def load_impl_params(step, id):
    """
    Load parameters that configure the implementation for a specific step.

    :param step: Dictionary containing step information.
    :param id: Configuration ID.
    :return: Dictionary of parameters.
    """
    logger.info(f"Loading parameters for step {step['id']}.")

    params = step['parameters']
    data = step['data']
    metrics = step['metrics']

    loader = data['loader']
    loader_module = load_module(name=loader)

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

            impl = load_module(name=cfg['implementation']['module'])
            params = load_impl_params(step, id)

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
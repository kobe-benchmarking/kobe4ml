import os
import importlib
import pandas as pd

from . import utils

logger = utils.get_logger(level='INFO')

def create_exp_dir(exp_name, dir):
    """
    Create experiment directory if it doesn't exist.

    :param exp_name: Name of the experiment.
    :param dir: Base directory for experiments.
    :return: Path to the experiment directory.
    """
    exp_dir = os.path.join(dir, exp_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        logger.info(f"Created experiment directory: {exp_dir}")
    else:
        print(".")
        logger.info(f"Experiment directory already exists: {exp_dir}")

    return exp_dir

def load_module(name):
    """
    Load a module dynamically using its name.

    :param name: Name of the module.
    :return: Loaded module.
    """
    module = importlib.import_module(name)
    logger.info(f"Module {name} loaded successfully.")

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

    loader = data['loader']
    loader_module = load_module(name=loader)

    ds_url = data['location']
    data_params = data['parameters']
    loader_params = {'url': ds_url}
    loader_params.update(data_params)

    dls = loader_module.preprocess(**loader_params)

    model_url = params['model_url']
    model_params = params['model']
    process_params = params['process']

    impl_params = {'pth': model_url, 'dls': dls}
    impl_params.update(model_params)
    impl_params.update(process_params)
    impl_params.update(metrics)

    return impl_params

def main(configs, dir='experiments'):
    """
    Main function to process configurations and run experiments.
    :param configs: List of configuration dictionaries.
    :param dir: Directory to save experiment results.
    """
    experiments_data = {}
    methods_dict = {"prepare": "train", "inference": "test"}

    for cfg in configs:
        logger.info(f"Reading configuration {cfg['metadata']['id']} for {cfg['metadata']['name']}.")
        
        exp_name = cfg['metadata']['parent_id']
        exp_dir = create_exp_dir(name=exp_name, dir=dir)

        if exp_name not in experiments_data:
            experiments_data[exp_name] = {
                "calls": [],
                "steps": [],
                "results": []
            }

        for step in cfg['step']:
            process = step['type']
            method = methods_dict[process]

            logger.info(f"Processing step {step['id']} for {method}ing benchmarking.")

            impl = load_module(name=cfg['implementation']['module'])
            params = load_impl_params(step)

            call = lambda: getattr(impl, method)(params)

            experiments_data[exp_name]["calls"].append(call)
            experiments_data[exp_name]["steps"].append(step['id'])

    for exp_name, data in experiments_data.items():
        for i, call in enumerate(data["calls"]):
            metrics = call()
            data["results"].append(metrics)

            logger.info(f"Metrics for step {i}: {metrics}.")

    for exp_name, data in experiments_data.items():
        if data["results"]:
            df = pd.DataFrame(data["results"])

            exp_dir = os.path.join(dir, exp_name)
            csv_path = os.path.join(exp_dir, "results.csv")

            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}.")

        else:
            logger.info("No results to save.")
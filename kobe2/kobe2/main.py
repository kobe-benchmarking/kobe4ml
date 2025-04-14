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

# def load_params(step, process):
#     logger.info(f"Loading parameters for step ID: {step['id']}")

#     model_url = step['model']['url']
#     ds_url = step['dataset']['url']

#     model_params = step['model']['parameters']
#     process_params = step['parameters']
#     batch_size = process_params["batch_size"]

#     loader = load_module(name=step['loader'])
#     dataloaders = getattr(loader, "preprocess")({"url": ds_url, 
#                                                  "batch_size": batch_size, 
#                                                  "process": process})

#     params = {'pth': model_url, 'dls': dataloaders}
#     params.update(model_params)
#     params.update(process_params)

#     return params

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


    #         params = load_params(step=step, process=process)

    #         call = lambda: getattr(impl, method)(params)

    #         experiments_data[exp_name]["calls"].append(call)
    #         experiments_data[exp_name]["steps"].append(step['id'])

    # for exp_name, data in experiments_data.items():
    #     for i, call in enumerate(data["calls"]):
    #         metrics = call()
    #         metrics["step"] = data["steps"][i]

    #         logger.info(f"[{exp_name}] Metrics for step {i}: {metrics}")

    #         data["results"].append(metrics)

    # for exp_name, data in experiments_data.items():
    #     if data["results"]:
    #         df = pd.DataFrame(data["results"])
    #         exp_dir = os.path.join(dir, exp_name)
    #         csv_path = os.path.join(exp_dir, "results.csv")
    #         df.to_csv(csv_path, index=False)

    #     logger.info(f"Results saved to {csv_path}")
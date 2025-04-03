import os
import importlib
import pandas as pd

from . import utils

logger = utils.get_logger(level='INFO')

def create_exp_dir(config, dir):
    exp_dir_name = config['metadata']['experiment']
    exp_dir = os.path.join(dir, exp_dir_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        logger.info(f"Created experiment directory: {exp_dir}")
    else:
        print(".")
        logger.info(f"Experiment directory already exists: {exp_dir}")

    return exp_dir

def load_module(name):
    module = importlib.import_module(name)
    logger.info(f"Module {name} loaded successfully")

    return module

def load_params(run, process):
    logger.info(f"Loading parameters for run ID: {run['id']}")

    model_url = run['model']['url']
    ds_url = run['dataset']['url']

    model_params = run['model']['parameters']
    process_params = run['parameters']

    dataloaders = preprocess(url=ds_url, params=process_params, process=process)

    params = {'pth': model_url, 'dls': dataloaders}
    params.update(model_params)
    params.update(process_params)

    return params

def main(experiments, dir='experiments'):
    experiments_data = {}

    for exp in experiments:
        logger.info(f"Running experiment for model: {exp['implementation']['module']}")
        
        exp_name = exp['metadata']['experiment']
        exp_dir = create_exp_dir(config=exp, dir=dir)

        if exp_name not in experiments_data:
            experiments_data[exp_name] = {
                "calls": [],
                "runs": [],
                "results": []
            }

        for run in exp['run']:
            logger.info(f"Processing run: {run['id']} with type: {run['type']}")

            process = run['type']
            method = "test" if process == 'inference' else "train"

            implementation = load_module(name=exp['implementation']['module'])
            params = load_params(run=run, process=process)

            call = lambda: getattr(implementation, method)(params)

            #loader = load_module(name=run['loader'])

            experiments_data[exp_name]["calls"].append(call)
            experiments_data[exp_name]["runs"].append(run['id'])

    for exp_name, data in experiments_data.items():
        for i, call in enumerate(data["calls"]):
            metrics = call()
            metrics["run"] = data["runs"][i]

            logger.info(f"[{exp_name}] Metrics for run {i}: {metrics}")

            data["results"].append(metrics)

    for exp_name, data in experiments_data.items():
        if data["results"]:
            df = pd.DataFrame(data["results"])
            exp_dir = os.path.join(dir, exp_name)
            csv_path = os.path.join(exp_dir, "results.csv")
            df.to_csv(csv_path, index=False)

        logger.info(f"Results saved to {csv_path}")
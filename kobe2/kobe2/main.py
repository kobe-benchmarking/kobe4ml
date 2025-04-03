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
    calls, runs, results = {}, {}, {}

    for exp in experiments:
        logger.info(f"Running experiment for model: {exp['implementation']['module']}")
        
        exp_name = exp['metadata']['experiment']
        exp_dir = create_exp_dir(config=exp, dir=dir)

        calls.setdefault(exp_name, [])
        runs.setdefault(exp_name, [])
        results.setdefault(exp_name, [])

        for run in exp['run']:
            logger.info(f"Processing run: {run['id']} with type: {run['type']}")

            process = run['type']
            method = "test" if process == 'inference' else "train"

            implementation = load_module(name=exp['implementation']['module'])
            params = load_params(run=run, process=process)

            call = lambda: getattr(implementation, method)(params)
            calls[exp_name].append(call)

            #loader = load_module(name=run['loader'])

            runs[exp_name].append(run['id'])

    for exp_name, call_list in calls.items():
        for i, call in enumerate(call_list):
            metrics = call()
            metrics["run"] = runs[exp_name][i] 

            logger.info(f"[{exp_name}] Metrics for run {i}: {metrics}")

            results[exp_name].append(metrics)

    for exp_name, metrics_list in results.items():
        if metrics_list:
            df = pd.DataFrame(metrics_list)
            exp_dir = os.path.join(dir, exp_name)
            csv_path = os.path.join(exp_dir, "results.csv")
            df.to_csv(csv_path, index=False)

        logger.info(f"Results saved to {csv_path}")
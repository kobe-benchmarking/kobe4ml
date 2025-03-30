import os
import sys
import importlib

from kobe2 import main as kobe
from . import utils
from .loader import *

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

def gather_configs(dir):
    logger.info(f"Gathering configurations from directory: {dir}")
    configs = []

    for file_name in os.listdir(dir):
        if file_name.endswith(".yaml"):
            file_path = os.path.join(dir, file_name)
            logger.info(f"Loading YAML file: {file_path}")

            configs.append(utils.load_yaml(file_path))

    return configs

def create_exp_dir(config, dir):
    exp_dir_name = config['metadata']['experiment']

    exp_dir = os.path.join(dir, exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"Created experiment directory: {exp_dir}")

    return exp_dir

def preprocess(url, batch_size):
    logger.info(f"Preprocessing data from URL: {url} with batch size: {batch_size}")

    samples, chunks = 7680, 32
    seq_len = samples // chunks

    bitbrain_dir = os.path.join(url, 'bitbrain')
    raw_dir = os.path.join(url, 'raw')

    get_boas_data(base_path=bitbrain_dir, output_path=raw_dir)
    datapaths = split_data(dir=raw_dir, train_size=3, val_size=2, test_size=2)
    _, _, test_df = get_dataframes(datapaths, seq_len=seq_len, exist=True)
    datasets = create_datasets(dataframes=(test_df,), seq_len=seq_len)
    dataloaders = create_dataloaders(datasets, batch_size=batch_size, drop_last=False)

    logger.info("Data preprocessing completed successfully")

    return dataloaders

def load_module(name, run):
    logger.info(f"Loading module: {name} for run ID: {run['id']}")

    model_url = run['model']['url']
    ds_url = run['dataset']['url']
    batch_size = run['parameters']['batch_size']

    model_params = run['model']['parameters']
    process_params = run['parameters']

    dataloaders = preprocess(url=ds_url, batch_size=batch_size)

    params = {'pth': model_url, 'dls': dataloaders}
    params.update(model_params)
    params.update(process_params)

    module = importlib.import_module(name)
    logger.info(f"Module {name} loaded successfully")

    return module, params

def main():
    logger.info("Starting experiment execution")

    experiments = gather_configs(dir='configs')
    calls, runs = [], []

    for exp in experiments:
        logger.info(f"Running experiment for model: {exp['implementation']['module']}")

        exp_dir = create_exp_dir(config=exp, dir='experiments')

        for run in exp['run']:
            logger.info(f"Processing run: {run['id']} with type: {run['type']}")

            process = run['type']
            method = "test" if process == 'inference' else "train"

            module, params = load_module(name=exp['implementation']['module'],
                                         run=run)

            call = lambda: getattr(module, method)(params)
            calls.append(call)

            runs.append(run['id'])

    logger.info("Starting Kobe for experiment runs")
    kobe(calls, runs, exp_dir, logger)

if __name__ == "__main__":
    main()
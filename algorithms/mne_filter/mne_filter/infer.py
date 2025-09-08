import warnings
import os
import numpy as np

from . import estimate
from . import utils
from .metrics import *
from .model import *

logger = utils.get_logger(level='DEBUG')

warnings.filterwarnings("ignore", category=FutureWarning)

def infer(data, model, metrics):
    """
    Test the model on the provided data and calculate the infer loss, MAE, and MSE.

    :param data: DataLoader for inference.
    :param model: The model to be tested.
    :param metrics: List of metric names to calculate (e.g., ['mae', 'mse']).
    :return: Dictionary containing metrics as defined in the input metrics list.
    """
    X, _ = data
    
    static_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'experiment', 'static', 'estims'))
    estims_path = utils.get_path(static_dir, filename="estims_mne.npy")

    X_flt = model(X)

    error = estimate.filter_error(X, X_flt)
    estims_array = np.stack([X, X_flt, error], axis=0)

    utils.save_np(data=estims_array, path=estims_path)
    logger.info(f"Saved per-sample errors to {estims_path}.")
        
    all_metrics = {
        'mae': mae(X, X_flt),
        'mse': mse(X, X_flt)
    }

    filtered_metrics = {metric: all_metrics[metric] for metric in metrics if metric in all_metrics}

    return filtered_metrics

def main(data_id, model, options):
    """
    Main function to execute the testing workflow.

    :param data_id: Dictionary containing dataset and its parameters.
    :param model: Path to the model JSON file containing model parameters.
    :param options: Dictionary containing process parameters and metrics to calculate.
    :return: Dictionary containing calculated metrics.
    """
    data, _ = data_id.values()
    model_params = utils.load_json(path=model)
    _, metrics = options.values()

    model = MNE_Filter(**model_params)
 
    results = infer(data=data[0],
                    model=model,
                    metrics=metrics)
    
    return results
import warnings
import zipfile
import os
import numpy as np

from . import utils
from .model import *

logger = utils.get_logger(level='DEBUG')

warnings.filterwarnings("ignore", category=FutureWarning)

def mae(X, X_dec):
    """
    Compute Mean Absolute Error (MAE) manually.
    
    :param X: Original data, ndarray of shape (n_samples, n_features)
    :param X_dec: Reconstructed/filtered data, same shape as X
    :return: Scalar MAE value
    """
    return np.mean(np.abs(X - X_dec))

def mse(X, X_dec):
    """
    Compute Mean Squared Error (MSE) manually.
    
    :param X: Original data, ndarray of shape (n_samples, n_features)
    :param X_dec: Reconstructed/filtered data, same shape as X
    :return: Scalar MSE value
    """
    return np.mean((X - X_dec)**2)

def unzip_model(path):
    """
    Extract a model zip file into the model weights and parameters for inference.

    :param path: Path to the zip file created by zip_model, e.g., models/attn_ae.zip
    :return: Tuple (model_params_dict, model_pth_path)
    """
    extract_dir = os.path.dirname(path)

    with zipfile.ZipFile(path, 'r') as zipf:
        zipf.extractall(extract_dir)

    pth_path = path.replace('.zip', '.pth')
    json_path = path.replace('.zip', '_params.json')

    model_params = utils.load_json(path=json_path)

    return model_params, pth_path

def infer(data, model, metrics):
    """
    Test the model on the provided data and calculate the test loss, MAE, and MSE.

    :param data: DataLoader for inference.
    :param model: The model to be tested.
    :param metrics: List of metric names to calculate (e.g., ['mae', 'mse']).
    :return: Dictionary containing metrics as defined in the input metrics list.
    """
    X, _ = data

    X_dec = model(X)
        
    all_metrics = {
        'mae': mae(X, X_dec),
        'mse': mse(X, X_dec)
    }

    filtered_metrics = {metric: all_metrics[metric] for metric in metrics if metric in all_metrics}

    return filtered_metrics

def main(data_id, model, options):
    """
    Main function to execute the testing workflow.

    :param data_id: Dictionary containing dataset and its parameters.
    :param model: Path to the model zip file containing model weights and parameters.
    :param options: Dictionary containing process parameters and metrics to calculate.
    :return: Dictionary containing calculated metrics.
    """
    data, _ = data_id.values()
    model_params = utils.load_json(path=model)
    _, metrics = options.values()

    model = MNE_Filter(**model_params)
 
    results = infer(data=data,
                    model=model,
                    metrics=metrics)
    
    return results
import torch
import zipfile
import os

from . import utils
from .model import *

logger = utils.get_logger(level='CRITICAL')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

def mae(X, X_dec):
    """
    Compute Mean Absolute Error (MAE) manually.

    :param X: Original input tensor.
    :param X_dec: Reconstructed output tensor.
    :return: MAE value.
    """
    return torch.mean(torch.abs(X - X_dec)).item()

def mse(X, X_dec):
    """
    Compute Mean Squared Error (MSE) manually.

    :param X: Original input tensor.
    :param X_dec: Reconstructed output tensor.
    :return: MSE value.
    """
    return torch.mean((X - X_dec) ** 2).item()

def unzip_model(path):
    """
    Extract a model zip file into the model weights and parameters for inference.

    :param path: Path to the zip file created by zip_model, e.g., models/conv_lstm_ae.zip
    :return: Tuple (model_pth_path, model_params_dict)
    """
    extract_dir = os.path.dirname(path)

    with zipfile.ZipFile(path, 'r') as zipf:
        zipf.extractall(extract_dir)

    model_pth = path.replace('.zip', '.pth')
    json_path = path.replace('.zip', '_params.json')

    model_params = utils.load_json(path=json_path)

    return model_pth, model_params

def infer(data, model, model_pth, metrics):
    """
    Test the model on the provided data and calculate the test loss, MAE, and MSE.

    :param data: Data to test the model on.
    :param model: The model to be evaluated.
    :param model_pth: Path to the pth file where the model weights are saved, e.g., models/conv_lstm_ae.pth.
    :param metrics: List of metric names to calculate (e.g., ['mae', 'mse']).
    :return: Dictionary containing metrics as defined in the input metrics list.
    """
    state_dict = utils.load_pth(path=model_pth)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    batches = len(data)

    total_infer_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0

    criterion = utils.BlendedLoss()

    with torch.no_grad():
        for _, (X, _, _) in enumerate(data):
            X = X.to(device)

            X_dec, _ = model(X)

            infer_loss = criterion(X_dec, X)
            total_infer_loss += infer_loss.item()

            total_mae += mae(X, X_dec)
            total_mse += mse(X, X_dec)

    avg_infer_loss = total_infer_loss / batches
    avg_mae = total_mae / batches
    avg_mse = total_mse / batches

    all_metrics = {
        'BlendedLoss': avg_infer_loss,
        'mae': avg_mae,
        'mse': avg_mse
    }

    filtered_metrics = {metric: all_metrics[metric] for metric in metrics if metric in all_metrics}

    return filtered_metrics

def main(params):
    """
    Main function to execute the testing workflow, including data preparation and model evaluation.
    """
    model_url, dls, metrics = params.values()

    model_pth, model_params = unzip_model(path=model_url)

    model = ConvLSTM_Autoencoder(**model_params)
 
    metrics = infer(data=dls[0],
                    model=model,
                    model_pth=model_pth,
                    metrics=metrics)
    
    return metrics
